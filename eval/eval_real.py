"""
Eval on real images from input/*_normalize.png, output to output/
"""
import sys
import os

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT_DIR, "src"))

import util
import torch
import numpy as np
from model import make_model
from render import NeRFRenderer
import torchvision.transforms as T
import torch.nn.functional as F
import tqdm
import imageio
from PIL import Image
from camera.loss import vgg_loss, clip_similarity

def extra_args(parser):
    parser.add_argument(
        "--input",
        "-I",
        type=str,
        default=os.path.join(ROOT_DIR, "input"),
        help="Image directory",
    )
    parser.add_argument(
        "--output",
        "-O",
        type=str,
        default=os.path.join(ROOT_DIR, "output"),
        help="Output directory",
    )
    parser.add_argument(
        "--filename",
        "-f",
        type=str,
        default=None,
        help="Specified file to process, if not None, ignore -I",
    )
    parser.add_argument("--size", type=int, default=64, help="Input image maxdim")
    parser.add_argument(
        "--out_size",
        type=str,
        default="64",
        help="Output image size, either 1 or 2 number (w h)",
    )

    parser.add_argument("--focal", type=float, default=119.4256, help="Focal length")

    parser.add_argument("--radius", type=float, default=-1, help="Camera distance") # 2.6
    parser.add_argument("--radius_m", type=float, default=2.0)
    parser.add_argument("--radius_M", type=float, default=5.0)
    parser.add_argument("--spacing", type=float, default=0.3)
    parser.add_argument("--z_near", type=float, default=0.8)
    parser.add_argument("--z_far", type=float, default=4.0)

    parser.add_argument(
        "--elevation",
        "-e",
        type=float,
        default=-10.0,
        help="Elevation angle (negative is above)",
    )
    parser.add_argument(
        "--num_views",
        type=int,
        default=40,
        help="Number of video frames (rotated views)",
    )
    parser.add_argument(
        "--scale", type=float, default=1.0, help="Video scale relative to input size"
    )
    parser.add_argument("--fps", type=int, default=30, help="FPS of video")
    parser.add_argument("--gif", action="store_true", help="Store gif instead of mp4")
    parser.add_argument(
        "--no_vid",
        action="store_true",
        help="Do not store video",
    )
    parser.add_argument(
        "--with_frame",
        action="store_true",
        help="Store frames",
    )
    return parser


args, conf = util.args.parse_args(
    extra_args, default_expname="srn_car", default_data_format="srn",
)
args.resume = True

device = util.get_cuda(args.gpu_id[0])
net = make_model(conf["model"]).to(device=device).load_weights(args)
renderer = NeRFRenderer.from_conf(
    conf["renderer"], eval_batch_size=args.ray_batch_size
).to(device=device)
render_par = renderer.bind_parallel(net, args.gpu_id, simple_output=True).eval()

z_near, z_far = args.z_near, args.z_far
focal = torch.tensor(args.focal, dtype=torch.float32, device=device)

in_sz = args.size
sz = list(map(int, args.out_size.split()))
if len(sz) == 1:
    H = W = sz[0]
else:
    assert len(sz) == 2
    W, H = sz

_coord_to_blender = util.coord_to_blender()
_coord_from_blender = util.coord_from_blender()

print("Generating rays")
# render_poses = torch.stack(
#     [
#         _coord_from_blender @ util.pose_spherical(angle, args.elevation, args.radius)
#         #  util.pose_spherical(angle, args.elevation, args.radius)
#         for angle in np.linspace(-180, 180, args.num_views + 1)[:-1]
#     ],
#     0,
# )  # (NV, 4, 4)

# render_rays = util.gen_rays(render_poses, W, H, focal, z_near, z_far).to(device=device)

if args.filename is None:
    inputs_all = os.listdir(args.input)
    inputs_all.sort()
    inputs = [
        os.path.join(args.input, x) for x in inputs_all # if x.endswith("_normalize.png")
    ]
else:
    inputs = [args.filename]
    
os.makedirs(args.output, exist_ok=True)

if len(inputs) == 0:
    if len(inputs_all) == 0:
        print("No input images found, please place an image into ./input")
    else:
        print("No processed input images found, did you run 'scripts/preproc.py'?")
    import sys

    sys.exit(1)
    
txt_path = os.path.join(args.output, "radius.txt")
with open(txt_path, "w") as f:
    pass

# cam_pose = torch.eye(4, device=device)
# cam_pose[2, -1] = args.radius
# print("SET DUMMY CAMERA")
# print(cam_pose)
# cam_pose = torch.tensor([[ 2.5882e-01, -4.8296e-01,  8.3652e-01,  2.2854e+00],
#          [ 9.6593e-01,  1.2941e-01, -2.2414e-01, -6.1236e-01],
#          [ 1.3869e-09,  8.6603e-01,  5.0000e-01,  1.3660e+00],
#          [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  1.0000e+00]], device=device)

# My version to regress the true camera extrinsics
# cam_pose = torch.eye(4, device=device)

image_to_tensor = util.get_image_to_tensor_balanced()

def render(radius, mode):
    cam_pose = torch.eye(4, device=device)
    cam_pose[2, -1] = radius
    print("SET DUMMY CAMERA")
    # print(cam_pose)
    # T = F.normalize(torch.Tensor([2.2854e+00, -6.1236e-01, 1.3660e+00]), dim=0) * radius
    # cam_pose = torch.tensor([[ 2.5882e-01, -4.8296e-01,  8.3652e-01,  T[0]], 
    #                          [ 9.6593e-01,  1.2941e-01, -2.2414e-01,  T[1]],
    #                          [ 1.3869e-09,  8.6603e-01,  5.0000e-01,  T[2]],
    #                          [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  1.0000e+00]], device=device)
    
    c = None
    if mode == 'sphere':
        render_poses = torch.stack(
                [
                    _coord_from_blender @ 
                    util.pose_spherical(angle, args.elevation, radius)
                    for angle in np.linspace(-180, 180, args.num_views + 1)[:-1]
                ],
                0,
            )  # (NV, 4, 4)
        NV = args.num_views
    elif mode == 'single':
        render_poses = cam_pose.unsqueeze(0)
        NV = 1

    render_rays = util.gen_rays(
        render_poses,
        W,
        H,
        focal * args.scale,
        z_near,
        z_far,
        c=c * args.scale if c is not None else None,
    ).to(device=device)
    
    net.encode(
        image.unsqueeze(0), cam_pose.unsqueeze(0), focal,
    )
    # print("Rendering", args.num_views * H * W, "rays")
    all_rgb_fine = []
    for rays in tqdm.tqdm(torch.split(render_rays.view(-1, 8), args.ray_batch_size, dim=0)):
        rgb, _depth = render_par(rays[None])
        all_rgb_fine.append(rgb[0])
    _depth = None
    rgb_fine = torch.cat(all_rgb_fine)
    frames = (rgb_fine.view(NV, H, W, 3).cpu().numpy() * 255).astype(
        np.uint8
    )
    return frames

with torch.no_grad():
    for i, image_path in enumerate(inputs):
        print("IMAGE", i + 1, "of", len(inputs), "@", image_path)
        image = Image.open(image_path).convert("RGB")
        image = T.Resize(in_sz)(image)
        image = image_to_tensor(image).to(device=device)
        
        radius_m = args.radius_m
        radius_M = args.radius_M
        spacing = args.spacing
        sample_num = int((radius_M - radius_m) / spacing)
        radiuses = [radius_m + j * spacing for j in range(sample_num)]
        
        loss_m = 1e10
        sim_m = -1
        best_radius = 2.6
        
        if args.radius == -1: # default, which means no specification
            for radius in radiuses:
                print("-------------------------------------------")
                print(f"=============> Now radius is {radius}")
                frames = render(radius=radius, mode='sphere')
                os.makedirs("tmp/", exist_ok=True)
                
                for i in range(args.num_views):
                    frm_path = os.path.join('tmp/', "{:04}.png".format(i))
                    imageio.imwrite(frm_path, frames[i])
                    
                sim = clip_similarity('tmp/', image_path, device)

                # if loss < loss_m:
                #     print("=============> New loss is", loss)
                #     print("=============> New best pose found!")
                #     best_radius = radius
                #     loss_m = loss

                if sim > sim_m:
                    print(f"=============> New sim is {sim}")
                    print(f"=============> New best radius {radius} found!")
                    best_radius = radius
                    sim_m = sim
                    
        else:
            best_radius = args.radius
        
        frames = render(best_radius, mode='sphere')

        im_name = os.path.basename(os.path.splitext(image_path)[0])

        if args.with_frame:
            frames_dir_name = os.path.join(args.output, im_name + "_frames")
            os.makedirs(frames_dir_name, exist_ok=True)

            for i in range(args.num_views):
                frm_path = os.path.join(frames_dir_name, "{:04}.png".format(i))
                imageio.imwrite(frm_path, frames[i])

        if not args.no_vid:
            if args.gif:
                vid_path = os.path.join(args.output, im_name + "_vid.gif")
                imageio.mimwrite(vid_path, frames, fps=args.fps)
            else:
                vid_path = os.path.join(args.output, im_name + "_vid.mp4")
                imageio.mimwrite(vid_path, frames, fps=args.fps, quality=8)
        print("Wrote to", vid_path)
        
        with open(txt_path, "a") as f:
            f.write(f"The best radius of {im_name} is {best_radius}\n")
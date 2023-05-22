import sys
import os

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
)

import torch
import torch.nn.functional as F
import numpy as np
import imageio
import util
import warnings
from data import get_split_dataset
from render import NeRFRenderer
from model import make_model
from scipy.interpolate import CubicSpline
import torchvision.transforms as T
import tqdm
import imageio
from PIL import Image
from camera.search import sample_pose_sphere
from camera.loss import pixel_loss
SAMPLE_POSE_NUM = 50


def extra_args(parser):
    parser.add_argument(
        "--subset", "-S", type=int, default=0, help="Subset in data to use"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Split of data to use train | val | test",
    )
    parser.add_argument(
        "--source",
        "-P",
        type=str,
        default="1",
        help="Source view(s) in image, in increasing order. -1 to do random",
    )
    parser.add_argument(
        "--num_views",
        type=int,
        default=40,
        help="Number of video frames (rotated views)",
    )
    parser.add_argument(
        "--elevation",
        type=float,
        default=-10.0,
        help="Elevation angle (negative is above)",
    )
    parser.add_argument(
        "--scale", type=float, default=1.0, help="Video scale relative to input size"
    )
    parser.add_argument(
        "--radius",
        type=float,
        default=0.0,
        help="Distance of camera from origin, default is average of z_far, z_near of dataset (only for non-DTU)",
    )
    parser.add_argument(
        "--img_path",
        type=str,
        default='my_input/IMG_4528_normalize.png',
        help="image path",
    )
    parser.add_argument("--fps", type=int, default=30, help="FPS of video")
    return parser


args, conf = util.args.parse_args(extra_args)
args.resume = True

device = util.get_cuda(args.gpu_id[0])

image_to_tensor = util.get_image_to_tensor_balanced()
images = imageio.imread(args.img_path)[..., :3]
images = image_to_tensor(images).unsqueeze(0).to(device=device)
print(images.shape)

poses = [torch.eye(4) for _ in range(int(args.source)+1)]
poses = torch.stack(poses, dim=0)
focal = 119.4256
if isinstance(focal, float):
    # Dataset implementations are not consistent about
    # returning float or scalar tensor in case of fx=fy
    focal = torch.tensor(focal, dtype=torch.float32)
focal = focal[None]

c = None
print(f"c is {c}")
if c is not None:
    c = c.to(device=device).unsqueeze(0)

NV, _, H, W = images.shape

if args.scale != 1.0:
    Ht = int(H * args.scale)
    Wt = int(W * args.scale)
    if abs(Ht / args.scale - H) > 1e-10 or abs(Wt / args.scale - W) > 1e-10:
        warnings.warn(
            "Inexact scaling, please check {} times ({}, {}) is integral".format(
                args.scale, H, W
            )
        )
    H, W = Ht, Wt

net = make_model(conf["model"]).to(device=device)
net.load_weights(args)

renderer = NeRFRenderer.from_conf(
    conf["renderer"], lindisp=False, eval_batch_size=args.ray_batch_size,
).to(device=device)

render_par = renderer.bind_parallel(net, args.gpu_id, simple_output=True).eval()

# Get the distance from camera to origin
z_near = 1.2
z_far = 4.0
print(f"z_near is {z_near}, z_far is {z_far}")

print("Generating rays")

dtu_format = False

if dtu_format:
    print("Using DTU camera trajectory")
    # Use hard-coded pose interpolation from IDR for DTU

    t_in = np.array([0, 2, 3, 5, 6]).astype(np.float32)
    pose_quat = torch.tensor(
        [
            [0.9698, 0.2121, 0.1203, -0.0039],
            [0.7020, 0.1578, 0.4525, 0.5268],
            [0.6766, 0.3176, 0.5179, 0.4161],
            [0.9085, 0.4020, 0.1139, -0.0025],
            [0.9698, 0.2121, 0.1203, -0.0039],
        ]
    )
    n_inter = args.num_views // 5
    args.num_views = n_inter * 5
    t_out = np.linspace(t_in[0], t_in[-1], n_inter * int(t_in[-1])).astype(np.float32)
    scales = np.array([2.0, 2.0, 2.0, 2.0, 2.0]).astype(np.float32)

    s_new = CubicSpline(t_in, scales, bc_type="periodic")
    s_new = s_new(t_out)

    q_new = CubicSpline(t_in, pose_quat.detach().cpu().numpy(), bc_type="periodic")
    q_new = q_new(t_out)
    q_new = q_new / np.linalg.norm(q_new, 2, 1)[:, None]
    q_new = torch.from_numpy(q_new).float()

    render_poses = []
    for i, (new_q, scale) in enumerate(zip(q_new, s_new)):
        new_q = new_q.unsqueeze(0)
        R = util.quat_to_rot(new_q)
        t = R[:, :, 2] * scale
        new_pose = torch.eye(4, dtype=torch.float32).unsqueeze(0)
        new_pose[:, :3, :3] = R
        new_pose[:, :3, 3] = t
        render_poses.append(new_pose)
    render_poses = torch.cat(render_poses, dim=0)
else:
    print("Using default (360 loop) camera trajectory")
    if args.radius == 0.0:
        radius = (z_near + z_far) * 0.5
        print("> Using default camera radius", radius)
    else:
        radius = args.radius

    # Use 360 pose sequence from NeRF
    # render_poses = torch.stack(
    #     [
    #         util.pose_spherical(angle, args.elevation, radius)
    #         for angle in np.linspace(-180, 180, args.num_views + 1)[:-1]
    #     ],
    #     0,
    # )  # (NV, 4, 4)
    render_poses = torch.tensor([[ 2.5882e-01, -4.8296e-01,  8.3652e-01,  2.2854e+00],
         [ 9.6593e-01,  1.2941e-01, -2.2414e-01, -6.1236e-01],
         [ 1.3869e-09,  8.6603e-01,  5.0000e-01,  1.3660e+00],
         [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  1.0000e+00]], device=device).unsqueeze(0)

render_rays = util.gen_rays(
    render_poses,
    W,
    H,
    focal * args.scale,
    z_near,
    z_far,
    c=c * args.scale if c is not None else None,
).to(device=device)
# (NV, H, W, 8)

focal = focal.to(device=device)

source = torch.tensor(list(map(int, args.source.split())), dtype=torch.long)
NS = len(source)
random_source = NS == 1 and source[0] == -1
# assert not (source >= NV).any()

if renderer.n_coarse < 64:
    # Ensure decent sampling resolution
    renderer.n_coarse = 64
    renderer.n_fine = 128

with torch.no_grad():
    print("Encoding source view(s)")
    if random_source:
        src_view = torch.randint(0, NV, (1,))
    else:
        src_view = source

    print(src_view)
    cam_pose = torch.eye(4, dtype=torch.float32)
    cam_pose[2, -1] = radius
    poses[src_view] = cam_pose
    
    cam_pose_baseline = torch.tensor([[ 2.5882e-01, -4.8296e-01,  8.3652e-01,  2.2854e+00],
         [ 9.6593e-01,  1.2941e-01, -2.2414e-01, -6.1236e-01],
         [ 1.3869e-09,  8.6603e-01,  5.0000e-01,  1.3660e+00],
         [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  1.0000e+00]], device=device)
    
    cam_pose_best = None # to be predicted
    loss_m = 1e10
    
    for _ in range(SAMPLE_POSE_NUM):
        cam_pose = sample_pose_sphere(radius).to(device)
        print("Reference pose\n", cam_pose_baseline)
        print("My pose\n", cam_pose)
    
        # image = Image.open(args.img_path).convert("RGB")
        # image = T.Resize(64)(image)
        # image = image_to_tensor(image).to(device=device)
        
        net.encode(
            images[0].unsqueeze(0),
            cam_pose.unsqueeze(0),
            focal,
            c=c,
        )
        
        # render one image

        print("Rendering", args.num_views * H * W, "rays")
        all_rgb_fine = []
        for rays in tqdm.tqdm(
            torch.split(render_rays.view(-1, 8), args.ray_batch_size, dim=0)
        ):
            rgb, _depth = render_par(rays[None])
            all_rgb_fine.append(rgb[0])
        _depth = None
        rgb_fine = torch.cat(all_rgb_fine)
        # rgb_fine (V*H*W, 3)

        frames = rgb_fine.view(-1, H, W, 3)
        
        pred_rgb = frames[0]
        gt_rgb = images[0].permute(1, 2, 0)
        print(pred_rgb.shape)
        print(gt_rgb.shape)
        loss = torch.nn.functional.mse_loss(pred_rgb, gt_rgb).item()
        if loss < loss_m:
            loss_m = loss
            cam_pose_best = cam_pose
            print("=============> New loss is", loss)
            print("=============> New best pose found!")
    
    # exit(0)

    # render with best pose
    cam_pose = cam_pose_best
    print("Reference pose\n", cam_pose_baseline)
    print("My pose\n", cam_pose)
    print("loss:", loss_m)

    net.encode(
        images[0].unsqueeze(0),
        cam_pose.unsqueeze(0),
        focal,
        c=c,
    )

    render_poses = torch.stack(
            [
                util.pose_spherical(angle, args.elevation, radius)
                for angle in np.linspace(-180, 180, args.num_views + 1)[:-1]
            ],
            0,
        )  # (NV, 4, 4)

    render_rays = util.gen_rays(
        render_poses,
        W,
        H,
        focal * args.scale,
        z_near,
        z_far,
        c=c * args.scale if c is not None else None,
    ).to(device=device)

    print("Rendering", args.num_views * H * W, "rays")
    all_rgb_fine = []
    for rays in tqdm.tqdm(
        torch.split(render_rays.view(-1, 8), args.ray_batch_size, dim=0)
    ):
        rgb, _depth = render_par(rays[None])
        all_rgb_fine.append(rgb[0])
    _depth = None
    rgb_fine = torch.cat(all_rgb_fine)
    # rgb_fine (V*H*W, 3)

    frames = rgb_fine.view(-1, H, W, 3)

print("Writing video")
# vid_name = "{:04}".format(args.subset)
vid_name = 'test'
if args.split == "test":
    vid_name = "t" + vid_name
elif args.split == "val":
    vid_name = "v" + vid_name
vid_name += "_v" + "_".join(map(lambda x: "{:03}".format(x), source))
vid_path = os.path.join(args.visual_path, args.name, "video" + vid_name + ".mp4")
viewimg_path = os.path.join(
    args.visual_path, args.name, "video" + vid_name + "_view.jpg"
)
imageio.mimwrite(
    vid_path, (frames.cpu().numpy() * 255).astype(np.uint8), fps=args.fps, quality=8
)

img_np = (images.permute(0, 2, 3, 1) * 0.5 + 0.5).cpu().numpy()
img_np = (img_np * 255).astype(np.uint8)
img_np = np.hstack((*img_np,))
imageio.imwrite(viewimg_path, img_np)

print("Wrote to", vid_path, "view:", viewimg_path)

ffmpeg -i ${1} -b 568k -r 30 -vf fps=30,scale=64:-1:flags=lanczos,palettegen -y tmp.png
ffmpeg -i ${1} -i tmp.png -r 30 -lavfi "fps=30,scale=64:-1:flags=lanczos[x];[x][1:v]paletteuse" -y ${2}
rm tmp.png

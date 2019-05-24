dir="./tmp"

for vid in `ls $dir`
do

    if [ -d ./tmp/$vid ]; then
        path=$dir/$vid
        ffmpeg -y -i $path/%d.jpg -r 10 $dir/$vid.mp4
    fi

done
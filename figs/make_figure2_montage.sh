convert figure2a.tiff -gravity northwest -weight bold -pointsize 80 -annotate 0 "A)" figure2a_ann.tiff
convert figure2b.tiff -gravity northwest -weight bold -pointsize 80 -annotate 0 "B)" figure2b_ann.tiff
montage figure2a_ann.tiff figure2b_ann.tiff -geometry +2+1 figure2.tiff

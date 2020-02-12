URL='https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz'
TAR_FILE=./data/oxford.tar.gz
OUTDIR=./data/oxford
mkdir -p $OUTDIR
wget -N $URL -O $TAR_FILE
tar xvzf $TAR_FILE -C $OUTDIR
rm $TAR_FILE

# Annotation data
URL='https://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz'
TAR_FILE=./data/oxford.annotations.tar.gz
wget -N $URL -O $TAR_FILE
tar xvzf $TAR_FILE -C $OUTDIR
rm $TAR_FILE
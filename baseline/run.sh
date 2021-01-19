SRCDIR=$(dirname "$0")
cd $SRCDIR

echo Training...
echo hyper parameter settings:
cat config.json
python3 main.py

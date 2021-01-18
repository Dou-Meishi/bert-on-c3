SRCDIR=$(dirname "$0")
cd $SRCDIR

echo prepare training/validation data
python3 prepare_data.py

echo Training...
echo hyper parameter settings:
cat bert_on_reshaped_c3.json
python3 bert_on_reshaped_c3.py

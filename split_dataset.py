import json
import random
from collections import defaultdict

TRAIN_PER_FAMILY = 1600
VAL_PER_FAMILY   = 200
TEST_PER_FAMILY  = 200
SEED             = 42

def split_dataset(input_path='data/dataset.jsonl', output_dir='data'):
    random.seed(SEED)

    # group records by family
    by_family = defaultdict(list)
    with open(input_path) as f:
        for line in f:
            line = line.strip()
            if line:
                record = json.loads(line)
                by_family[record['family']].append(record)

    train, val, test = [], [], []

    for family, records in by_family.items():
        random.shuffle(records)
        n = len(records)
        need = TRAIN_PER_FAMILY + VAL_PER_FAMILY + TEST_PER_FAMILY
        if n < need:
            raise ValueError(f"{family} has {n} records but need {need}")

        train += records[:TRAIN_PER_FAMILY]
        val   += records[TRAIN_PER_FAMILY : TRAIN_PER_FAMILY + VAL_PER_FAMILY]
        test  += records[TRAIN_PER_FAMILY + VAL_PER_FAMILY : need]

        print(f"{family}: {TRAIN_PER_FAMILY} train / {VAL_PER_FAMILY} val / {TEST_PER_FAMILY} test")

    # shuffle so families are interleaved (important for training)
    random.shuffle(train)
    random.shuffle(val)
    random.shuffle(test)

    def write(records, path):
        with open(path, 'w') as f:
            for r in records:
                f.write(json.dumps(r) + '\n')
        print(f"Wrote {len(records)} records to {path}")

    write(train, f"{output_dir}/train.jsonl")
    write(val,   f"{output_dir}/val.jsonl")
    write(test,  f"{output_dir}/test.jsonl")

if __name__ == '__main__':
    split_dataset()

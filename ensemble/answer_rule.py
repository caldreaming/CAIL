import json
# "投保的人是谁", "投保人是谁",
target_question = ["投保的人是谁", "投保人是谁"]
punc = [',', '.', '?', '!', '，', '。', '！', '？']


def extract_question(origin_file):
    count = 0
    with open(origin_file, 'r', encoding="utf-8") as f:
        train_data = json.load(f)["data"]
        questions = []
        for i, example in enumerate(train_data):
            # print("第{}个样本".format(i))

            doc = example["paragraphs"][0]["context"]
            qas = example["paragraphs"][0]["qas"]
            # print(doc)
            # print(qas)
            domain = example["domain"]
            casename = example["paragraphs"][0]["casename"]
            for qa in qas:
                q = qa["question"]
                a = ""
                if len(qa["answers"]) > 0:
                    a = qa["answers"][0]["text"]
                qid = qa["id"]
                for ti in target_question:
                    if ti in q:
                        questions.append({"question": q, "answer": a, "casename": casename, "doc": doc})
    return questions


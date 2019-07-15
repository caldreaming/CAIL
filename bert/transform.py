import json

origin_file = "cail_base/predictions.json"
result_file = "result.json"
with open(result_file, 'w', encoding="utf-8") as g:
    with open(origin_file, 'r', encoding="utf-8") as f:
        preds = json.load(f)
        pred_dict = []
        for id in preds.keys():
            answer = preds[id]
            pred_dict.append({"answer": answer, "id": id})
        json.dump(pred_dict, g, ensure_ascii=False)
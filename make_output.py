import json
import pickle
class ResultRecord:
    def __init__(self,start,end):
        self.start=start
        self.end=end
        self.cache={}
        for i in range(start,end+1):
            self.cache[i]=None

    def reset(self):
        for i in range(self.start, self.end + 1):
            self.cache[i] = None

    def record(self,ids,scores):
        for i in range(len(ids)):
            self.cache[ids[i]]=float(scores[i])

    def export(self,path='./results.json'):
        stream=[]
        for i in range(self.start, self.end + 1):
            stream.append({'video':'%d.mp4'%i,'predict_score': self.cache[i]})
        with open(path, "w", encoding="utf-8") as json_file:
            json.dump(stream, json_file)


if __name__=='__main__':
    box = []
    rd=ResultRecord(27223,34029)
    for i in range(5):
        with open('./results/results/ep10_%d.pkl'%i,'rb') as f:
            b=pickle.load(f)
            box.append(b)

    for k in b.keys():
        v = (box[0][k] + box[1][k] + box[2][k] + box[3][k] + box[4][k]) / 5
        rd.record([k], [v])

    rd.export('output.json')
from utils import *
from rouge import Rouge
import textstat
from tqdm import tqdm

def avg_rouge_score(data: list, lay_sum: list):
    if(len(data) != len(lay_sum)):
        print(f"The num of pred_sen is not equal to ans_sen!")
        return
    rouge = Rouge()
    final_rouge_list = [0, 0, 0]
    for i in tqdm(range(len(data))):
        rouge_score = rouge.get_scores(data[i].lower(), lay_sum[i].lower())
        r_1, r_2, r_l = (rouge_score[0]['rouge-1']['f'], 
                rouge_score[0]['rouge-2']['f'], rouge_score[0]['rouge-l']['f'])
        final_rouge_list[0] += r_1
        final_rouge_list[1] += r_2
        final_rouge_list[2] += r_l

    final_rouge_list = [score*100 for score in final_rouge_list]
    return [round((x/len(data)), 4) for x in final_rouge_list]

def avg_read_score(dataset: list):
    fkgl_score = 0
    dcrs_score = 0
    for data in tqdm(dataset):
        fkgl_score += textstat.flesch_kincaid_grade(data)
        dcrs_score += textstat.dale_chall_readability_score(data)

    return (fkgl_score/len(dataset), dcrs_score/len(dataset))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default='PLOS')
    parser.add_argument("--datatype", type=str, default="val")
    args = parser.parse_args()
    
    article, lay_sum, keyword = load_task1_data(args)
    file_path = './bart_plos_2.txt'
    data = get_data(file_path)
    
    # setting the num of pred_data and laysum
    data = data
    lay_sum = lay_sum[:5]

    # avg rouge
    final_rouge = avg_rouge_score(data, lay_sum)
    print("rouge_score:")
    print(final_rouge)

    # avg readability
    fkgl_score, dcrs_score = avg_read_score(data)
    print(f"FkGL: {round(fkgl_score, 4)}")
    print(f"DCRS: {round(dcrs_score, 4)}")

    

        
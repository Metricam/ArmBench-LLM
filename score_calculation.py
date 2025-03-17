from label_extraction_patterns import extract_answer_for_numeric_choices, extract_again_for_numeric_choices, extract_final_for_numeric_choices
from label_extraction_patterns import extract_answer_for_numeric_answer, extract_again_for_numeric_answer, extract_final_for_numeric_answer
from label_extraction_patterns import extract_answer_for_letter_choices, extract_again_for_letter_choices, extract_final_for_letter_choices
from label_extraction_patterns import extract_correct_answers_list, extract_correct_answers_dict
import json
import os
import pandas as pd

def calculate_scores_for_mmlu(outputs_file, df, model_name):
    with open(outputs_file, 'r') as f:
        batch_results = json.load(f)

    labels = []
    if df.shape[0] > len(batch_results):
        raise ValueError("Evaluation is not complete for mmlu-pro-hy")

    for result in batch_results:
        text = result['output']
        true_answer = result['label']
        extracted_answer = extract_answer_for_letter_choices(text)

        if extracted_answer != None and extracted_answer == true_answer:
            labels.append(True)
        elif extracted_answer != None and extracted_answer != true_answer:
            labels.append(False)
        elif extracted_answer == None:
                extracted_answer = extract_again_for_letter_choices(text)  
                if extracted_answer != None and extracted_answer == true_answer:
                    labels.append(True)
                elif extracted_answer != None and extracted_answer != true_answer:
                    labels.append(False)
                elif extracted_answer == None:
                    extracted_answer = extract_again_for_letter_choices(text)  
                    if extracted_answer != None and extracted_answer == true_answer:
                        labels.append(True)
                    elif extracted_answer != None and extracted_answer != true_answer:
                        labels.append(False)
                    else:
                        labels.append(False)
    df = pd.concat([df, pd.DataFrame(labels, columns=['prediction'])], axis=1)

    result_df = df.groupby("category")["prediction"].mean().reset_index()
    keys = result_df['category'].tolist()
    keys = [col.title() for col in keys]  
    keys = ['Model', 'Average'] + keys

    values = result_df['prediction'].tolist()
    values = [model_name, sum(values) / len(values)] + values  

    result_df = pd.DataFrame([values], columns=keys)

    result_dict = result_df.iloc[0].to_dict()

    return result_dict

def calculate_scores_for_exam_tests(outputs_file, df,  model_name):
    with open(outputs_file, 'r') as f:
        batch_results = json.load(f)
    score = 0
    if df.shape[0] > len(batch_results):
        raise ValueError(f"Evaluation is not complete for {outputs_file.split('/')[-1].split('_')[0]}")

    for idx, result in enumerate(batch_results):
        task_type = df.iloc[idx]['task_type']
        text = result['output']
        true_answer = result['label']

        if task_type == 1:
            extracted_answer = extract_answer_for_numeric_choices(text)
            if extracted_answer != None and extracted_answer==true_answer[0]:
                score += 0.25
            else:
                extracted_answer = extract_again_for_numeric_choices(text)
                if extracted_answer != None and extracted_answer==true_answer[0]:
                    score += 0.25
                else:
                    extracted_answer = extract_final_for_numeric_choices(text)
                    if extracted_answer != None and extracted_answer==true_answer[0]:
                        score += 0.25
                    else:
                        pass

        elif task_type == 2:
            extracted_answer = extract_correct_answers_list(text)
            extracted_answer = [str(i) for i in extracted_answer]
            if extracted_answer != None and set(true_answer) == set(extracted_answer):
             score += 0.25

        elif task_type == 3:
            extracted_answer = extract_correct_answers_list(text)
            chsy_score = 0
            for a,b in zip(extracted_answer, true_answer):
                if a==b:
                    chsy_score+=0.25
                elif a!=b:
                    chsy_score-=0.25
                elif a=='Չգիտեմ':
                    pass
            if chsy_score < 0:
                chsy_score = 0
            score+=chsy_score
        
        elif task_type == 4:
            extracted_answer = extract_correct_answers_dict(text)
            extracted_answer = [str(i) for i in extracted_answer]

            if extracted_answer == true_answer:
                score+=0.25

        elif task_type == 5:
            extracted_answer = extract_correct_answers_list(text)
            extracted_answer = [str(i) for i in extracted_answer]

            if extracted_answer == true_answer:
                score+=0.25

        elif task_type == 6:
            extracted_answer = extract_answer_for_letter_choices(text)
            if extracted_answer != None and extracted_answer==true_answer[0]:
                score += 0.25
            else:
                extracted_answer = extract_again_for_letter_choices(text)
                if extracted_answer != None and extracted_answer==true_answer[0]:
                    score += 0.25
                else:
                    extracted_answer = extract_final_for_letter_choices(text)
                    if extracted_answer != None and extracted_answer==true_answer[0]:
                        score += 0.25
                    else:
                        pass
        elif task_type == 7:
            extracted_answer = extract_answer_for_numeric_answer(text)
            if extracted_answer != None and extracted_answer==true_answer[0]:
                score += 0.25
            else:
                extracted_answer = extract_again_for_numeric_answer(text)
                if extracted_answer != None and extracted_answer==true_answer[0]:
                    score += 0.25
                else:
                    extracted_answer = extract_final_for_numeric_answer(text)
                    if extracted_answer != None and extracted_answer==true_answer[0]:
                        score += 0.25
                    else:
                        pass

    return {
        'model': model_name,
        'score': score
    }

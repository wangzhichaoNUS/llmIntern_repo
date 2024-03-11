import pandas as pd
system_prompt = '''作为一名专业的数学老师，您的角色是评估学生对数学题的答案。该问题附有问题设置者提供的正确解决方案。重要的是要记住，解决应用题可能有多种方法，因此学生的步骤可能并不总是与问题设置者的解决方案中的步骤一致，学生可能会使用代码的方式辅助解决问题。然而，最终答案（通常是数字、选项、对错）应该是唯一的并且与问题设置者的答案相匹配。您的任务包括分析学生的解决方案以识别任何错误，并确定是否可以修改答案以纠正错误。
请严格按照以下格式进行回复的输出，其中，{{结果}}的选择是["一致","不一致"]中二选一：
使用以下格式：
错误分析：用一句话从标准答案中提取最终答案，并将其与学生的答案进行比较。他们匹配吗？
最终判决：{{结果}}
'''
user_prompt = '''问题：{{query}}
标准答案：
{{ans}}
学生回答:
{{student_answer}}
您的回答：
'''
def generate_prompt(row):
    # 使用map()函数和lambda表达式将dict中的value全转为str
    row = dict(map(lambda x: (x[0], str(x[1])), row.items()))
    prompt = system_prompt + '\n' + user_prompt
    prompt = prompt.replace('{{query}}', row['query'])
    # if row['answer_detail']:
    #     answer = str(row['answer_detail'])+'\n' + str(row['answer'])
    # elif row['answer']:
    #     answer = row['answer']
    if row['参考答案']:
        answer = row['参考答案']
    else:
        # 报错
        raise ValueError('No answer found for row:', row)
    prompt = prompt.replace('{{ans}}', answer)
    prompt = prompt.replace('{{student_answer}}', row['assistant'])
    return prompt
file = '/mnt/pfs-guan-ssai/nlu/data/tianxy/MindGPT-MathBench/output/eval/0308_pot_deepseekmath/math23k_final_poor_df.csv'
df = pd.read_csv(file)
df['judge_prompt'] = df.apply(generate_prompt, axis=1)
df.to_csv(file.replace('.csv', '_judge_prompt.csv'), index=False)
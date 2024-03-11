import argparse
import numpy as np
import pandas as pd
import os
from tqdm import tqdm
from typing import Generator, List
# openai 最新版
from openai import OpenAI
from typing_extensions import Required,NotRequired,TypedDict
from pprint import pprint
from collections.abc import Iterable
import threading

from concurrent.futures import ThreadPoolExecutor,as_completed




class GenerateConfig(TypedDict):
    model:Required[str]
    temperature:NotRequired[float]
    top_p: NotRequired[float]
    max_tokens:Required[int]
    n:NotRequired[int]
    stop:Required[str]



def get_args():
    parser = argparse.ArgumentParser(description='information')
    parser.add_argument("--input_f", type=str, default="/mnt/pfs-guan-ssai/nlu/data/tianxy/MindGPT-MathBench/output/eval/0308_pot_deepseekmath/math23k_final_poor_df_judge_prompt.csv", help="data path")
    parser.add_argument("--output_dir", type=str,
                        default="/mnt/pfs-guan-ssai/nlu/data/tianxy/MindGPT-MathBench/output/eval/0308_pot_deepseekmath/cmath_final_poor_df_judge_prompt_updated.csv", help="output data path")
    parser.add_argument("--url", type=str,
                        default="http://172.24.138.127:8000/v1",
                        # 172.24.138.127:8000
                        # 172.24.136.32:8002
                        help="openai api url")
    parser.add_argument("--columns",type=str,
                        default='judge_prompt', help="columns of q")
    parser.add_argument("--ans_columns", type=str,
                        default='judge_res', help="columns of a")
    parser.add_argument("--thread_num", type=int,default=30,help="thread num")
    parser.add_argument("--chunk_num", type=int,
                        default=1, help="chunk num")
    parser.add_argument("--max_tokens", type=int,default=2048,help="max generated tokens")
    args = parser.parse_args()
    return args


def complete_response(messages, stream=False, **kwargs):
    if stream:
        completion = client.chat.completions.create(
            messages=messages,
            stream=stream,
            **kwargs
        )
        for chunk in completion:
            # print(chunk)
            if hasattr(chunk.choices[0].delta, "content") and chunk.choices[0].delta.content != None:
                yield chunk.choices[0].delta.content
    else:
        try:
            completion = client.chat.completions.create(
                messages=messages, **kwargs)
            yield completion.choices[0].message.content
        except Exception as e:
            print('='*25)
            print(e)
            print('='*25)
            yield '<|worry data|>'

def task(row:dict):
    data = [
        {"role": "system",
            "content": "You are a helpful assistant."},
        {"role": "user",
            "content": row[args.columns]},
    ]

    response = complete_response(
        messages=data, stream=False, **generate_config)
    if isinstance(response, Generator):
        response = ''.join(response)
    else:
        response= '<|worry data|>'

    row[args.ans_columns] = response
    return row

def get_df(input_f:str):
    if input_f.endswith(".csv"):
        df = pd.read_csv(input_f)
    elif input_f.endswith(".json"):
        df = pd.read_json(input_f)
    elif input_f.endswith(".jsonl"):
        df = pd.read_json(input_f, lines=True)
    else:
        raise ValueError('input file format is not supported')
    print(f'文件总行数:{len(df)}')
    return df



def chunk_dataframe(df, chunk_num: int):

    total_rows = df.shape[0]
    print("chunk size is", int(total_rows / chunk_num))
    assert int(total_rows / chunk_num) >0, 'chunk size is too small'
    # 获取每个chunk块对应的index
    chunks = np.array_split(range(total_rows), chunk_num)
    for idx, current_chunk in zip(range(chunk_num), chunks):
        current_chunk = current_chunk.tolist()
        # 根据每个chunk块的第一个和最后一个index切分
        chunk_df = df[current_chunk[0]: (current_chunk[-1] + 1)]
        print(f'chunk_{idx},range({current_chunk[0]},{current_chunk[-1]})')
        yield chunk_df


def format_time(seconds):
    if seconds < 60:
        return f"{seconds:.2f} seconds"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.2f} minutes"
    else:
        hours = seconds / 3600
        return f"{hours:.2f} hours"

def get_response_pool(file: str,chunk_num:int, thread_num: int):
    import time
    start_time = time.time()
    df = get_df(file)
    chunk_generator = chunk_dataframe(df, chunk_num)

    # 创建一个线程池，最大线程数为..
    dfs = []
    with ThreadPoolExecutor(max_workers=thread_num) as executor:
        for idx,chunk_df in enumerate(chunk_generator):
            chunk_start_time = time.time()
            # 每个chunk块使用n个线程去推理
            results = executor.map(task, chunk_df.to_dict('records')) # 每行转为一个字典
            # 遍历结果
            final_results = []
            for result in tqdm(results,total=len(chunk_df)):
                final_results.append(result)
            # 保存结果
            final_df = pd.DataFrame(final_results)
            dfs.append(final_df)
            final_df.to_csv(os.path.join(save_path, f'part_{idx}.csv'), index=False)
            print(
                f'part_{idx} done! time cost: {format_time(time.time()-chunk_start_time)}')
        # 删除tmp文件
        [os.remove(os.path.join(save_path, f'part_{idx}.csv')) for idx in range(chunk_num)]
        df_merge = pd.concat(dfs)
        _,filename = os.path.split(file)
        df_merge.to_csv(os.path.join(save_path, f'{filename}_all.csv'), index=False)

    print(f'所有线程完成! time cost: {format_time(time.time()-start_time)}')


def file_process(file:str,prompt_file:str):
    df = get_df(file)[:]
    with open(prompt_file, 'r') as f:
        prompt = f.read()
        df['prompts'] = df['instruction'].apply(
            lambda x: prompt.replace('{{query}}', x))
    df.to_csv(f'{file}_add_prompt.csv', index=False)
    print('添加prompt完毕！')

if __name__ == "__main__":
    args = get_args()
    print(args)
    base_url = args.url  # qwen
    api_key = 'none'

    # 获取当前脚本的目录路径
    current_directory_path = os.path.dirname(os.path.abspath(__file__))
    print("当前脚本所在的目录路径:", current_directory_path)

    client = OpenAI(base_url=base_url, api_key=api_key, max_retries=2)
    generate_config: GenerateConfig = {"model": "qwen",
                                       "temperature": 0.9,
                                    #    "top_p": 0.8,
                                       "max_tokens": args.max_tokens,
                                       "stop": ['<|endoftext|>'],
                                       }
    test_col = args.columns
    file = args.input_f
    save_path = args.output_dir
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    # get_response_thread(file, args.thread_num)
    # 如果是文件夹
    if os.path.isdir(file):
        for root, dirs, files in os.walk(file):
            for f in files:
                get_response_pool(f, args.chunk_num, args.thread_num)
    else:
        get_response_pool(file, args.chunk_num, args.thread_num)
import numpy as np
from vllm import LLM, SamplingParams

### templates
template_w_ref = "Please act as a precise judge and evaluate the quality of the answer to question. Rate the answer from 0 to 9, where a higher value means a better answer. Please refer to the reference answer to make your judgement. Respond with an integer between 0 and 9. \n\n[Question]\n{instruction}\n\n[Reference]\n{reference}\n\n[Answer]\n{input}"

template_wo_ref = "Please act as a precise judge and evaluate the quality of the answer to question. Rate the answer from 0 to 9, where a higher value means a better answer. Please respond with an integer between 0 and 9. \n\n[Question]\n{instruction}\n\n[Answer]\n{input}"


def evaluate(prompts):
    sampling_params = SamplingParams(temperature=0,
                                     top_p=1,
                                     top_k=-1, 
                                     max_tokens=10,
                                     stop = '</s>',
                                     presence_penalty=1.2,
                                     logprobs=20,
                                     use_beam_search=True,
                                     best_of=2
                                     )

    outputs = model.generate(prompts, sampling_params)
    outputs = [ (int(output.request_id), output) for output in outputs]
    sorted_list = sorted(outputs, key=lambda x: x[0])
    outputs = [x[1] for x in sorted_list]

    scores = []

    for output in outputs:
        logprobs = output.outputs[0].logprobs
        class_ids = [29900, 29896, 29906, 29941, 29946, 29945, 29953, 29955, 29947, 29929]
        class_values = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

        try:
            class_probs = []
            for class_id in class_ids:
                temp_list = [v for k, v in logprobs[0].items() if class_id == k]
                if len(temp_list) == 0:
                    class_probs.append(0)
                else:
                    class_probs.append( np.exp(temp_list[0]) )
            class_probs = np.array(class_probs) / np.sum(class_probs) 
            score = np.sum(class_probs * np.array(class_values))  
        except Exception as e:
            print(f'{e}')
            score = 0.
        scores.append(score)
    return scores


### load the model
path_to_model=''
model = LLM(model=path_to_model)

### test example
question = ''
response = ''
reference = '' ## it will be reference-based evaluation if the reference is given

## reference-based evaluation
prompt = template_w_ref.format(instruction=question, reference=reference, input=response) ## form the data sample

## reference-free evaluation
prompt = template_wo_ref.format(instruction=question, input=response)  ## form the data sample


## generate the score
score = evaluate([prompt])
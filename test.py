import subprocess
import time
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('test_size', type=str, help='whether to use small or large question test set', default="small")
args = parser.parse_args()

if args.test_size == "small":
    questions = [
        "What's natural language processing?",
        "List some benefits of using \"virtual nodes\" in AWS Dynamo.",
    ]
else:
    questions = [
        "What's natural language processing?",
        "List some benefits of using \"virtual nodes\" in AWS Dynamo.",
        "How to store credential in Git?",
        "What are ways for feature selection in data science?",
        "How to convert a continuous feature into categorical feature?",
        "What are some popular tools for text mining and how do they work?",
        "What are the differences between supervised and unsupervised machine learning algorithms?",
        "What is the purpose of cross-validation in machine learning?",
        "How can deep learning be used for image recognition?",
        "What is the difference between a convolutional neural network and a recurrent neural network?",
        "What are some common preprocessing techniques used in natural language processing?",
        "What is the difference between overfitting and underfitting in machine learning?",
        "What are some common algorithms used for anomaly detection?",
        "What is the difference between precision and recall in machine learning evaluation metrics?",
        "How can feature scaling improve the performance of machine learning models?",
        "What is transfer learning and how can it be applied in machine learning?",
        "What are some common methods for dimensionality reduction in data science?",
        "How can imbalanced data affect the performance of machine learning models?",
        "What are some common methods for text classification in natural language processing?",
        "What is the purpose of regularization in machine learning?"
    ]

n = len(questions)
total_runtime = 0

for question in questions:
    print(f"==================================================")
    print(f"Testing question: {question}")
    command = ["python", "qa.py", question]

    start_time = time.monotonic()
    result = subprocess.run(command, capture_output=True)
    end_time = time.monotonic()

    if result.returncode == 0:
        output = result.stdout.decode().strip()
        print(f"{output}")
    else:
        error = result.stderr.decode().strip()
        print(f"Error: {error}")
    
    runtime = end_time - start_time
    total_runtime += runtime
    print(f"Runtime: {runtime:.2f} seconds")

average_runtime = total_runtime / n
print(f"Average runtime over {n} questions: {average_runtime:.2f} seconds")

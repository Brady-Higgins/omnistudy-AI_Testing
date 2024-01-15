from docStore import context_QA

def main():
    pass
    QA = context_QA()
    QA.initQA(index_name = 'haystack',top_k=2)
    user_input=""
    while user_input!="Exit":
        user_input = input("Question: ")
        output=QA.QA_output(query=user_input)
        print(output[0])


if __name__=="__main__":
    main()

from AI_QA import QA
from Process_PDF import FileUpload

def main():
    # f = FileUpload()
    # path = ".\Textbooks\CrackingTheCodingInterview.pdf"
    # textbook_name="CrackingTheCodingInterview"
    # f.upload(path=path,textbook_name=textbook_name)
    # print("Complete")

    Bart = QA()
    Bart.init_QA(index_name = 'haystack',namespace= 'CrackingTheCodingInterview')
    user_input=""
    while user_input!="Exit":
        user_input = input("Question: ")
        if user_input=="Exit":
            print("It depends on what you mean by exit. If you mean exit in the sense that you\'re leaving the building, then yes, it\'s possible to exit the building. However, if you\'re referring to the exit of the building itself, then it\'s not possible. If you\'re talking about the exit from the building as a whole, then there\'s no such thing as an exit.")
            break
        context = Bart.retrieve_docs(query=user_input)
        print(context)

        output=Bart.run(query=user_input)
        print(output)


if __name__=="__main__":
    main()

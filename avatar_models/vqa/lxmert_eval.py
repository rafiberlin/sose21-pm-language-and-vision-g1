from avatar_models.vqa.lxmert.lxmert import LXMERTInference

# Just an example how to use the LXMERT Inference
if __name__ == "__main__":

    #URL = "https://www.wallpapers13.com/wp-content/uploads/2015/12/Nature-Lake-Bled.-Desktop-background-image.jpg"
    #URL = "https://vignette.wikia.nocookie.net/spongebob/images/2/20/SpongeBob's_pineapple_house_in_Season_7-4.png/revision/latest/scale-to-width-down/639?cb=20151213202515"
    URL = "https://www.quizible.com/sites/quiz/files/imagecache/question/quiz/pictures/2012/06/14/q40716.jpg"
    test_question = "which animal is this?"
    lxmert = LXMERTInference()
    answer = lxmert.infer(URL, test_question)
    print("Question:", test_question)
    print("Answer:", answer)

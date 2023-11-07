import googletrans
import pandas as pd

class Translation():
    def __init__(self, from_lang='vi', to_lang='en'):
        # The class Translation is a wrapper for the two translation libraries, googletrans and translate. 
        self.__to_lang = to_lang
        self.translator = googletrans.Translator()

    def preprocessing(self, text):
        """
        It takes a string as input, and returns a string with all the letters in lowercase

        :param text: The text to be processed
        :return: The text is being returned in lowercase.
        """
        return text.lower()

    def __call__(self, text):
        """
        The function takes in a text and preprocesses it before translation

        :param text: The text to be translated
        :return: The translated text.
        """
        text = self.preprocessing(text)
        return self.translator.translate(text, dest=self.__to_lang).text
    
if __name__ == '__main__':
    translator = Translation()

    train_data = pd.read_csv("./data/train/info.csv")
    test_data = pd.read_csv("./data/test/info.csv")

    for i in range(len(train_data)):
        train_data.loc[i, "caption"] = translator(train_data.loc[i, "caption"])
        train_data.loc[i, "description"] = translator(train_data.loc[i, "description"])
        train_data.loc[i, "moreInfo"] = translator(train_data.loc[i, "moreInfo"])

    for i in range(len(test_data)):
        test_data.loc[i, "caption"] = translator(test_data.loc[i, "caption"])
        test_data.loc[i, "description"] = translator(test_data.loc[i, "description"])
        test_data.loc[i, "moreInfo"] = translator(test_data.loc[i, "moreInfo"])
    
    train_data.to_csv('./data/train/info_trans.csv')
    test_data.to_csv('./data/test/info_trans.csv')
import re
import ftfy
from legalnlp.mask_functions import *


def clean_bert(text):
    """
         Cleans a text based on bad Unicode and other characters

         Parameters
         -----------
         texto:  str
             A piece of text


         Returns
         -----------
         str
             Fixed text

     """

    txt = ftfy.fix_text(text)
    txt = txt.replace("\n", " ")
    txt = re.sub(' +', ' ', txt)
    return(txt)


def clean(text, lower=True, return_masked=False):
    """
        Cleans a text by removing general patterns, such as url, email, acronyms and other symbols, plural
        of words and specific Portuguese-related grammar

        Parameters
        -----------
        texto:  str
            A piece of text

        lower: bool
            Whether to lowercase text (Default: True)

        return_masked: bool
            If return_masked == False, the function outputs a clean text. Otherwise, it returns a dictionary containing the clean text and the information extracted by RegEx (Default: False)


        Returns
        -----------
        dict or str
     

    """

    dic = {}

    # Limpeza geral
    dic['txt'], dic['url'] = mask_url(text)  # Remove URLs
    dic['txt'], dic['email'] = mask_email(dic['txt'])  # Remove emails
    # Siglas (e.g., C.P.F => CPF)
    dic['txt'] = re.sub("([A-Z])\.", r"\1", dic['txt'])
    if lower:
        dic['txt'] = dic['txt'].lower()  # Tornando letras minúsculas
    dic['txt'] = re.sub("s[\/\.]a", " sa ", dic['txt'],
                        flags=re.I)  # s.a ou s/a => sa
    dic['txt'] = dic['txt'].replace(" - - ", " - ")
    dic['txt'] = dic['txt'].replace(" - ", " - - ")
    # Colocando espaço aos lados dos símbolos
    dic['txt'] = re.sub("(\W)", r" \1 ", dic['txt'])
    dic['txt'] = dic['txt'].replace("\n", " ")
    dic['txt'] = dic['txt'].replace("\t", " ")

    # Possíveis plurais e gênero
    dic['txt'] = dic['txt'].replace("( s )", "(s)")
    dic['txt'] = dic['txt'].replace("( a )", "(a)")
    dic['txt'] = dic['txt'].replace("( as )", "(as)")
    dic['txt'] = dic['txt'].replace("( o )", "(o)")
    dic['txt'] = dic['txt'].replace("( os )", "(os)")

    # Juntando algumas strings
    dic['txt'] = re.sub("(?<=\d) [-\.] (?=\d)", '', dic['txt'])
    dic['txt'] = re.sub("(?<=\d) , (?=\d)", ',', dic['txt'])
    dic['txt'] = dic['txt'].replace("[ email ]", "[email]")
    dic['txt'] = dic['txt'].replace("[ url ]", "[url]")
    # (e.g., arquivem - se => arquivem-se)
    dic['txt'] = re.sub("(\w) - (\w)", r"\1-\2", dic['txt'])
    dic['txt'] = re.sub(' +', ' ', dic['txt'])

    # Mascarando
    dic['txt'], dic['oab'] = mask_oab(dic['txt'])
    dic['txt'], dic['data'] = mask_data(dic['txt'])
    dic['txt'], dic['processo'] = mask_processo(dic['txt'])
    # Consideramos que as casas decimais são dadas pela vírgula
    dic['txt'], dic['valor'] = mask_valor(dic['txt'])
    dic['txt'], dic['numero'] = mask_numero(dic['txt'])

    # Extra spaces
    dic['txt'] = re.sub(' +', ' ', dic['txt'])
    dic['txt'] = dic['txt'].strip()

    # Output
    if return_masked:
        return dic
    else:
        return dic['txt']

import re

def mask_email(txt):

    """
        Finds an email pattern and then masks it.

        Parameters
        -----------
        txt:  str
            A piece of text containing the email pattern


        Returns
        -----------
        str
            masked email string as ' [email] '

        list
            list with the found pattern(s)

    """

    
    pattern=r'[^\s]+@[^\s]+'
    sub=' [email] '

    return re.sub(pattern, sub, txt, flags=re.I), re.findall(pattern, txt, flags=re.I)

def mask_url(txt):

    """
        Finds an url pattern and then masks it.

        Parameters
        -----------
        txt:  str
            A piece of text containing the url pattern


        Returns
        -----------
        str
            masked url string as ' [url] '

        list
            list with the found pattern(s)

    """
        
    pattern='http\S+'
    pattern2='www\S+'
    sub=' [url] '

    txt, find = re.sub(pattern, sub, txt, flags=re.I), re.findall(pattern, txt, flags=re.I)
    txt, find2 = re.sub(pattern2, sub, txt, flags=re.I), re.findall(pattern2, txt, flags=re.I)

    return txt, find+find2

def mask_oab(txt):

    """
        Finds an OAB (which stands for Order of Attorneys of Brazil) pattern and then masks it.

        Parameters
        -----------
        txt:  str
            A piece of text containing the OAB pattern


        Returns
        -----------
        str
            masked OAB string as ' [oab] '

        list
            list with the found pattern(s)

    """
    
    find=[]
    pattern='OAB\s?[:-]?\s?\d+\s?/?\s?[A-Z]?[A-Z]?'
    pattern2='OAB\s?/?\s?[A-Z]?[A-Z]?\s?[:-]?\s?\d+'
    sub=' [oab] '

    txt, find = re.sub(pattern, sub, txt, flags=re.I), re.findall(pattern, txt, flags=re.I)
    txt, find2 = re.sub(pattern2, sub, txt, flags=re.I), re.findall(pattern2, txt, flags=re.I)

    return txt, find+find2

def mask_data(txt):

    """
        Finds a date-format pattern and then masks it.

        Parameters
        -----------
        txt:  str
            A piece of text containing the date


        Returns
        -----------
        str
            masked date string as ' [data] '

        list
            list with the found pattern(s)

    """

    
    pattern="\d{2}\s?\/\s?\d{2}\s?\/\s?\d{4}"
    sub=' [data] '

    return re.sub(pattern, sub, txt, flags=re.I), re.findall(pattern, txt, flags=re.I)

def mask_processo(txt, num=15):

    """
        Finds a lawsuit number pattern and then masks it.

        Parameters
        -----------
        txt:  str
            A piece of text containing the lawsuit number pattern


        Returns
        -----------
        str
            masked lawsuit number string as ' [processo] '

        list
            list with the found pattern(s)

    """
        
    pattern="\d{"+str(num)+",}" #consideramos números com mais de 15 dígitos como sendo o número de um processo
    sub=' [processo] '

    return re.sub(pattern, sub, txt, flags=re.I), re.findall(pattern, txt, flags=re.I)

def mask_numero(txt):

    """
        Finds a number pattern and then masks it.

        Parameters
        -----------
        txt:  str
            A piece of text containing the number pattern


        Returns
        -----------
        str
            masked number string as ' [numero] '

        list
            list with the found pattern(s)
            
    """
        
    pattern="\d+"
    sub=' [numero] '

    return re.sub(pattern, sub, txt, flags=re.I), re.findall(pattern, txt, flags=re.I)

def mask_valor(txt):

    """
        Finds a value pattern and then masks it.

        Parameters
        -----------
        txt:  str
            A piece of text containing the value pattern


        Returns
        -----------
        str
            masked value string as ' [valor] '

        list
            list with the found pattern(s)

    """
        

    pattern="R\s?\$\s?\d+[.,]?\d+[.,]?\d+"
    sub=' [valor] '

    return re.sub(pattern, sub, txt, flags=re.I), re.findall(pattern, txt, flags=re.I)

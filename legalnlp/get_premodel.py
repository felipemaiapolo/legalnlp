import wget
import zipfile


def get_premodel(model):
    modelv = False
    d = None
    if model == 'bert':
        # BERTikal
        url = 'https://ndownloader.figshare.com/files/30446754'
        filename = wget.download(url, out=d)
        if d == None:
            d = ''
        with zipfile.ZipFile(d+filename, "r") as zip_ref:
            zip_ref.extractall(d+filename.replace('.zip', ''))
        modelv = True
    # Download files to use in Word2Vec and Doc2Vec
    if model == 'wodc':
        url2 = 'https://ndownloader.figshare.com/files/30446736'
        filename2 = wget.download(url2, out=d)
        if d == None:
            d = ''
        with zipfile.ZipFile(d+filename2, "r") as zip_ref:
            zip_ref.extractall(d+filename2.replace('.zip', ''))
        modelv = True
    
    # Download Word2Vec of NILC
    if model == 'w2vnilc':
        url2 = 'http://143.107.183.175:22980/download.php?file=embeddings/word2vec/cbow_s100.zip'
        filename2 = wget.download(url2, out=d)
        if d == None:
            d = ''
        with zipfile.ZipFile(d+filename2, "r") as zip_ref:
            zip_ref.extractall(d+filename2.replace('.zip', ''))
        modelv = True 
    # Download files to use Phraser model
    if model == 'phraser':
        url2 = 'https://ndownloader.figshare.com/files/30446727'
        filename2 = wget.download(url2, out=d)
        if d == None:
            d = ''
        with zipfile.ZipFile(d+filename2, "r") as zip_ref:
            zip_ref.extractall(d+filename2.replace('.zip', ''))
        modelv = True
    # Download files to use Fast Text model
    if model == 'fasttext':
        url2 = 'https://ndownloader.figshare.com/files/30446739'
        filename2 = wget.download(url2, out=d)
        if d == None:
            d = ''
        with zipfile.ZipFile(d+filename2, "r") as zip_ref:
            zip_ref.extractall(d+filename2.replace('.zip', ''))
        modelv = True
    # Download files to use NeuralMind pre-model base
    if model == 'neuralmindbase':
        url2 = 'https://neuralmind-ai.s3.us-east-2.amazonaws.com/nlp/bert-base-portuguese-cased/bert-base-portuguese-cased_pytorch_checkpoint.zip'
        url_vocab = 'https://neuralmind-ai.s3.us-east-2.amazonaws.com/nlp/bert-base-portuguese-cased/vocab.txt'
        filename2 = wget.download(url2, out=d)
        filename3 = wget.download(url_vocab, out=d)
        if d == None:
            d = ''
        with zipfile.ZipFile(d+filename2, "r") as zip_ref:
            zip_ref.extractall(d+filename2.replace('.zip', ''))
        modelv = True
    # Download files to use NeuralMind pre-model large
    if model == 'neuralmindlarge':
        url2 = 'https://neuralmind-ai.s3.us-east-2.amazonaws.com/nlp/bert-large-portuguese-cased/bert-large-portuguese-cased_pytorch_checkpoint.zip'
        url_vocab = 'https://neuralmind-ai.s3.us-east-2.amazonaws.com/nlp/bert-large-portuguese-cased/vocab.txt'
        filename2 = wget.download(url2, out=d)
        filename3 = wget.download(url_vocab, out=d)
        if d == None:
            d = ''
        with zipfile.ZipFile(d+filename2, "r") as zip_ref:
            zip_ref.extractall(d+filename2.replace('.zip', ''))
        modelv = True
    # If don't download any model return false, else return true
    return modelv

import re

#this function to replace non ascii with empty string
def replace_trash(unicode_string):
    for i in range(0, len(unicode_string)):
        try:
            unicode_string[i].encode("ascii")
            replaced = False
        except:
            unicode_string=''
            replaced = True
    return replaced, unicode_string

def clearstring(string):
    string = re.sub('[^A-Za-z0-9 ]+', '', string)
    string = string.split(' ')
    string = filter(None, string)
    string = [y.strip() for y in string]
    string = ' '.join(string)
    return string

def separate_dataset(trainset):
    datastring = []
    datatarget = []
    for i in xrange(len(trainset.data)):
        data_ = trainset.data[i].split('\n')
        data_ = filter(None, data_)
        for n in xrange(len(data_)):
		    data_[n] = clearstring(data_[n])
        datastring += data_
        for n in xrange(len(data_)):
            datatarget.append(trainset.target[i])
    return datastring, datatarget
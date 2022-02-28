def log(sthg, mode):
    if mode=="r":
        msg = "\033[01;31m{0}\033[00m".format(str(sthg))
    elif mode=="g":
        msg = "\033[01;32m{0}\033[00m".format(str(sthg))
    elif mode=="b":
        msg = "\033[01;34m{0}\033[00m".format(str(sthg))
    elif mode=="y":
        msg = "\033[01;93m{0}\033[00m".format(str(sthg))
    print (msg)

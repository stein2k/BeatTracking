import os
from scipy.io import wavfile

def main(filename):

    if not os.path.exists(filename):
        raise RuntimeError("Could not find audio file %s" % filename)

    audio_data = wavfile.read(filename)

if __name__ == '__main__':
    from optparse import OptionParser
    parser = OptionParser()
    (options,args) = parser.parse_args()

    if len(args) != 1:
        raise RuntimeError("Must specify a single input file")

    main(args[0])

import os
import errno


def get_dirs(base_path):
    return sorted([f for f in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, f))])


def get_files(base_path,extension,append_base=False):
    if (append_base):
        files =[os.path.join(base_path,f) for f in os.listdir(base_path) if (f.endswith(extension) and not f.startswith('.'))]
    else:
        files = [f for f in os.listdir(base_path) if (f.endswith(extension) and not f.startswith('.'))]
    return sorted(files)



def txt_read(path):
    with open(path) as f:
        content = f.readlines()
    lines = [x.strip() for x in content]
    return lines

def txt_write(path,lines,mode='w'):
    out_file = open(path, mode)
    for line in lines:
        out_file.write(line)
        out_file.write('\n')
    out_file.close()


def touch_dir(path):
    if(not os.path.exists(path)):
        os.makedirs(path)

def touch_file_dir(file_path):
    if not os.path.exists(os.path.dirname(file_path)):
        try:
            os.makedirs(os.path.dirname(file_path))
        except OSError as exc:  # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise





def get_file_name_ext(inputFilepath):
    filename_w_ext = os.path.basename(inputFilepath)
    filename, file_extension = os.path.splitext(filename_w_ext)
    return filename, file_extension

def get_latest_file(path,extension=''):
    files = get_files(path,extension=extension,append_base=True)
    return max(files, key=os.path.getctime)

def dir_empty(path):
    if os.listdir(path) == []:
        return True
    else:
        return False

def chkpt_exists(path):
    files = [f for f in os.listdir(path) if (f.find('.ckpt') > 0 and not f.startswith('.'))]
    if len(files):
        return True
    return False

def ask_yes_no_question(question):
    print(question+ ' [y/n] ')
    while True:
        answer = input()
        if answer.lower() in ['y','yes']:
            return True
        elif answer.lower() in ['n','no']:
            return False
        print('Please Enter a valid answer')

def file_size(file):
    return os.path.getsize(file)
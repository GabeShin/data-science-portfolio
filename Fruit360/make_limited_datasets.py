# https://deeplearningcourses.com/c/advanced-computer-vision
# https://www.udemy.com/advanced-computer-vision
import os

def mkdir(p):
      if not os.path.exists(p):
        os.mkdir(p)

def link(src, dst):
    if not os.path.exists(dst):
        os.symlink(src, dst, target_is_directory=True)
    
def getLimitedDataset():
    mkdir('../input/fruits-360-small')
    
    classes = [
      'Apple Golden 1',
      'Avocado',
      'Lemon',
      'Mango',
      'Kiwi',
      'Banana',
      'Strawberry',
      'Raspberry'
    ]

    train_path_from = os.path.abspath('../input/fruits-360/Training')
    valid_path_from = os.path.abspath('../input/fruits-360/Test')

    train_path_to = os.path.abspath('../input/fruits-360-small/Training')
    valid_path_to = os.path.abspath('../input/fruits-360-small/Validation')

    mkdir(train_path_to)
    mkdir(valid_path_to)


    for c in classes:
      link(train_path_from + '/' + c, train_path_to + '/' + c)
      link(valid_path_from + '/' + c, valid_path_to + '/' + c)
        
def main():
    getLimitedDataset()

if __name__ == '__main__':
    main()    
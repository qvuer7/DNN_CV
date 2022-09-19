from torchvision import transforms
from PIL import Image


def transform(size, mean, std):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(size),
        transforms.Normalize(mean = mean , std = std)
    ])
    return transform

def renormalize(mean, std):
    mean_1 = list(map(lambda x: -x[0]/x[1], zip(mean, std)))
    std_1 = list(map(lambda x: 1/x, std))

    transform= transforms.Normalize(
        mean=mean_1,
        std=std_1
    )
    return transform



def test():

    im_pil = Image.open('/Users/andriizelenko/qvuer7/projects/NN_CV_ptorch/2022-08-05 15.12.24.jpg')
    mean = (0.5, 0.5, 0.5)
    std =  (0.5, 0.5, 0.5)

    t2 = transforms.Compose([transforms.ToPILImage()])
    t = transform(size = (224,224), mean = mean, std = std)
    t_ren = renormalize(mean, std)


    im_t = t(im_pil)

    im_ren = t_ren(im_t)

    im_pil = t2(im_ren)
    im_pil.show()

if __name__ == '__main__':
    test()

'''
mean = [ 0.485, 0.456, 0.406 ],
std = [ 0.229, 0.224, 0.225 ]

mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.255],
        std=[1 / 0.229, 1 / 0.224, 1 / 0.255]
'''



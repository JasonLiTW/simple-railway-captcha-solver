from PIL import Image, ImageDraw, ImageFont
from random import randint
import csv
import numpy as np
FONTPATH = ["./data/font/times-bold.ttf", "./data/font/courier-bold.ttf"]
ENGSTR = "ABCDEFGHJKLMNPQRSTUVWXYZ" # 沒有O和I
LETTERSTR = "0123456789ABCDEFGHJKLMNPQRSTUVWXYZ"


class rect:
    def __init__(self):
        self.size = (randint(5, 21), randint(5, 21))
        self.location = (randint(1, 199), randint(1, 59))
        self.luoverlay = True if randint(1, 10) > 6 else False
        self.rdoverlay = False if self.luoverlay else True if randint(1, 10) > 8 else False
        self.lucolor = 0 if randint(0, 1) else 255
        self.rdcolor = 0 if self.lucolor == 255 else 255
        self.ludrawn = False
        self.rddrawn = False
        self.pattern = randint(0, 1)


    def draw(self, image, overlay):
        if((overlay or not self.luoverlay) and not self.ludrawn):
            self.ludrawn = True
            stp = self.location
            transparent = int(255 * 0.45 if self.lucolor == 0 else 255 * 0.8)
            color = (self.lucolor, self.lucolor, self.lucolor, transparent)
            uline = Image.new("RGBA", (self.size[0], 1), color)
            lline = Image.new("RGBA", (1, self.size[1]), color)
            image.paste(uline, stp, uline)
            image.paste(lline, stp, lline)
        if((overlay or not self.rdoverlay) and not self.rddrawn):
            self.rddrawn = True
            dstp = (self.location[0], self.location[1] + self.size[1])
            rstp = (self.location[0] + self.size[0], self.location[1])
            transparent = int(255 * 0.45 if self.rdcolor == 0 else 255 * 0.8)
            color = (self.rdcolor, self.rdcolor, self.rdcolor, transparent)
            dline = Image.new("RGBA", (self.size[0], 1), color)
            rline = Image.new("RGBA", (1, self.size[1]), color)
            image.paste(dline, dstp, dline)
            image.paste(rline, rstp, rline)


class captchatext:
    def __init__(self, priority, offset, captchalen, engletter, ENGNOLIMIT):
        self.engletter = engletter
        if ENGNOLIMIT:
            engletter = True if randint(1, 34) <= 24 else False
        if engletter:
            self.letter = ENGSTR[randint(0, len(ENGSTR) - 1)]
        else:
            self.letter = str(randint(0, 9))
        self.color = [randint(10, 140) for _ in range(3)]
        self.angle = randint(-55, 55)
        self.priority = priority
        self.offset = offset
        self.next_offset = 0
        self.captchalen = captchalen


    def draw(self, image):
        color = (self.color[0], self.color[1], self.color[2], 255)
        font = ImageFont.truetype(FONTPATH[randint(0, 1)], randint(25, 27) * 10)
        text = Image.new("RGBA", (font.getsize(self.letter)[0], 300), (0, 0, 0, 0))
        textdraw = ImageDraw.Draw(text)
        textdraw.text((0, 0), self.letter, font=font, fill=color)
        text = text.rotate(self.angle, expand=True)
        text = text.resize((int(text.size[0] / 10), int(text.size[1] / 10)))
        base = int(self.priority * (200 / self.captchalen))
        rand_min = (self.offset - base - 4) if (self.offset - base - 4) >= -15 else -15
        rand_min = 0 if self.priority == 0 else rand_min
        avg_dp = int(200 / self.captchalen)
        rand_max = (avg_dp - text.size[0]) if self.priority == self.captchalen - 1 else (avg_dp - text.size[0] + 10)
        try:
            displace = randint(rand_min, rand_max)
        except:
            displace = rand_max
        location = (base + displace, randint(3, 23))
        self.next_offset = location[0] + text.size[0]
        image.paste(text, location, text)


def generate(GENNUM, SAVEPATH, ENGP=25, FIVEP=0, ENGNOLIMIT=False, filename="train"):
    captchacsv = open(SAVEPATH + "captcha_{:s}.csv".format(filename), 'w', encoding = 'utf8', newline = '')
    lencsv = open(SAVEPATH + "len_{:s}.csv".format(filename), 'w', encoding = 'utf8', newline = '')
    letterlist = []
    lenlist = []
    for index in range(1, GENNUM + 1, 1):
        captchastr = ""
        captchalen = 5 if randint(1, 100) <= FIVEP else 6
        engat = randint(0, captchalen - 1) if randint(1, 100) <= ENGP else -1
        bgcolor = [randint(180, 250) for _ in range(3)]
        captcha = Image.new('RGBA', (200, 60), (bgcolor[0], bgcolor[1], bgcolor[2], 255))
        rectlist = [rect() for _ in range(32)]
        for obj in rectlist:
            obj.draw(image=captcha, overlay=False)
        offset = 0
        for i in range(captchalen):
            newtext = captchatext(i, offset, captchalen, (True if engat == i else False), ENGNOLIMIT)
            newtext.draw(image=captcha)
            offset = newtext.next_offset
            captchastr += str(newtext.letter)
        letterlist.append([str(index), captchastr])
        lenlist.append([str(index), captchalen])
        for obj in rectlist:
            obj.draw(image=captcha, overlay=True)
        captcha.convert("RGB").save(SAVEPATH + str(index).zfill(len(str(GENNUM))) + ".jpg", "JPEG")
    writer = csv.writer(captchacsv)
    writer.writerows(letterlist)
    writer = csv.writer(lencsv)
    writer.writerows(lenlist)
    captchacsv.close()
    lencsv.close()


if __name__ == "__main__":
    generate(50000, "./data/56_imitate_train_set/",  ENGP=100, FIVEP=50, ENGNOLIMIT=True, filename="train")
    generate(10240, "./data/56_imitate_vali_set/",  ENGP=100, FIVEP=50, ENGNOLIMIT=True, filename="vali")
    generate(50000, "./data/5_imitate_train_set/",  ENGP=100, FIVEP=100, ENGNOLIMIT=True, filename="train")
    generate(10240, "./data/5_imitate_vali_set/",  ENGP=100, FIVEP=100, ENGNOLIMIT=True, filename="vali")
    generate(50000, "./data/6_imitate_train_set/",  ENGP=100, FIVEP=0, ENGNOLIMIT=True, filename="train")
    generate(10240, "./data/6_imitate_vali_set/",  ENGP=100, FIVEP=0, ENGNOLIMIT=True, filename="vali")

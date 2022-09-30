
class Base:
    def __init__(self, x, y, img):
        self.x = x
        self.y = y
        self.img = img
        self.vel = 3

    def move(self):
        self.x -= self.vel

    def draw(self, win):
        win.blit(self.img, (self.x, self.y))

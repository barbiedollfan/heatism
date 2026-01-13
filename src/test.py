class Wallet:
    def __init__(self, money):
        self.money = money

account = Wallet(100)
account.credit = 800
print(hasattr(account, 'credit'))

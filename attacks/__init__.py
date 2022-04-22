from attacks.blended import Blended

TRIGGERS = {
    'blended': Blended,
}


def build_trigger(attack_name, img_size, num, mode, target, trigger):
    assert attack_name in TRIGGERS.keys()
    trigger = TRIGGERS[attack_name](img_size, num, mode, target, trigger)
    return trigger

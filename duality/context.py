from collections import namedtuple
DIR='/Users/morino/Downloads/sentiment_anaysis/tweet140/data';
PORT=13000;

# use this dictionary to select the local name of the action
action_name_dict={
    0: "NONO", # which means give no action response
    1: "wave",
    2: "sqat",
    3: "stand",
    4: "turn right",
    5: "turn left",
    6: "walk forward",
    7: "stop walking",
    8: "walk backward",
    9: "look left",
    10: "look right",
    11: "jump",
    12: "goup",
    13: "godown",
    14: "angry",
    15: "blink",
    16: "smile",
    17: "wink",
    18: "wink_R",
    19: "close><",
    20: "whiteeye",
    21:  "a",
    22: "i",
    23: "u",
    24: "o",
    25: "regret",
    26: "box",
    27: "cheek",
    28: "cheekB",
    29: "tear",
    30: "clear"
}

voice_name_dict= {
    0: "recognition_error",
    1: "positive_comment",
    2: "negtive_comment",
    3: "neutral_comment",
    4: "permission"
}


song_name_dict={
    12: "Elysion",
    13: "my time",
    14: "love2-4-11",
    15: "WinterAlice",
    16: "relations",
    17: "Bad Apple"
}



WORD_VEC_SIZE=300;



NOISE_LEVEL=0.0005;

ACTION_POOL_SIZE=31;

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'));


# define 200 as sing dance
SING_DANCE_NO=63;

DEFAULT_OPTION_NO=-1;

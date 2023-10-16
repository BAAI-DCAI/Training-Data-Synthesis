import os
import random
import numpy as np
from collections import defaultdict

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, ConcatDataset
from torchvision.datasets import ImageNet

from PIL import Image
import json

imagenet_classes = ["tench", "goldfish", "great white shark", "tiger shark", "hammerhead shark", "electric ray",
                        "stingray", "rooster", "hen", "ostrich", "brambling", "goldfinch", "house finch", "junco",
                        "indigo bunting", "American robin", "bulbul", "jay", "magpie", "chickadee", "American dipper",
                        "kite (bird of prey)", "bald eagle", "vulture", "great grey owl", "fire salamander",
                        "smooth newt", "newt", "spotted salamander", "axolotl", "American bullfrog", "tree frog",
                        "tailed frog", "loggerhead sea turtle", "leatherback sea turtle", "mud turtle", "terrapin",
                        "box turtle", "banded gecko", "green iguana", "Carolina anole",
                        "desert grassland whiptail lizard", "agama", "frilled-necked lizard", "alligator lizard",
                        "Gila monster", "European green lizard", "chameleon", "Komodo dragon", "Nile crocodile",
                        "American alligator", "triceratops", "worm snake", "ring-necked snake",
                        "eastern hog-nosed snake", "smooth green snake", "kingsnake", "garter snake", "water snake",
                        "vine snake", "night snake", "boa constrictor", "African rock python", "Indian cobra",
                        "green mamba", "sea snake", "Saharan horned viper", "eastern diamondback rattlesnake",
                        "sidewinder rattlesnake", "trilobite", "harvestman", "scorpion", "yellow garden spider",
                        "barn spider", "European garden spider", "southern black widow", "tarantula", "wolf spider",
                        "tick", "centipede", "black grouse", "ptarmigan", "ruffed grouse", "prairie grouse", "peafowl",
                        "quail", "partridge", "african grey parrot", "macaw", "sulphur-crested cockatoo", "lorikeet",
                        "coucal", "bee eater", "hornbill", "hummingbird", "jacamar", "toucan", "duck",
                        "red-breasted merganser", "goose", "black swan", "tusker", "echidna", "platypus", "wallaby",
                        "koala", "wombat", "jellyfish", "sea anemone", "brain coral", "flatworm", "nematode", "conch",
                        "snail", "slug", "sea slug", "chiton", "chambered nautilus", "Dungeness crab", "rock crab",
                        "fiddler crab", "red king crab", "American lobster", "spiny lobster", "crayfish", "hermit crab",
                        "isopod", "white stork", "black stork", "spoonbill", "flamingo", "little blue heron",
                        "great egret", "bittern bird", "crane bird", "limpkin", "common gallinule", "American coot",
                        "bustard", "ruddy turnstone", "dunlin", "common redshank", "dowitcher", "oystercatcher",
                        "pelican", "king penguin", "albatross", "grey whale", "killer whale", "dugong", "sea lion",
                        "Chihuahua", "Japanese Chin", "Maltese", "Pekingese", "Shih Tzu", "King Charles Spaniel",
                        "Papillon", "toy terrier", "Rhodesian Ridgeback", "Afghan Hound", "Basset Hound", "Beagle",
                        "Bloodhound", "Bluetick Coonhound", "Black and Tan Coonhound", "Treeing Walker Coonhound",
                        "English foxhound", "Redbone Coonhound", "borzoi", "Irish Wolfhound", "Italian Greyhound",
                        "Whippet", "Ibizan Hound", "Norwegian Elkhound", "Otterhound", "Saluki", "Scottish Deerhound",
                        "Weimaraner", "Staffordshire Bull Terrier", "American Staffordshire Terrier",
                        "Bedlington Terrier", "Border Terrier", "Kerry Blue Terrier", "Irish Terrier",
                        "Norfolk Terrier", "Norwich Terrier", "Yorkshire Terrier", "Wire Fox Terrier",
                        "Lakeland Terrier", "Sealyham Terrier", "Airedale Terrier", "Cairn Terrier",
                        "Australian Terrier", "Dandie Dinmont Terrier", "Boston Terrier", "Miniature Schnauzer",
                        "Giant Schnauzer", "Standard Schnauzer", "Scottish Terrier", "Tibetan Terrier",
                        "Australian Silky Terrier", "Soft-coated Wheaten Terrier", "West Highland White Terrier",
                        "Lhasa Apso", "Flat-Coated Retriever", "Curly-coated Retriever", "Golden Retriever",
                        "Labrador Retriever", "Chesapeake Bay Retriever", "German Shorthaired Pointer", "Vizsla",
                        "English Setter", "Irish Setter", "Gordon Setter", "Brittany dog", "Clumber Spaniel",
                        "English Springer Spaniel", "Welsh Springer Spaniel", "Cocker Spaniel", "Sussex Spaniel",
                        "Irish Water Spaniel", "Kuvasz", "Schipperke", "Groenendael dog", "Malinois", "Briard",
                        "Australian Kelpie", "Komondor", "Old English Sheepdog", "Shetland Sheepdog", "collie",
                        "Border Collie", "Bouvier des Flandres dog", "Rottweiler", "German Shepherd Dog", "Dobermann",
                        "Miniature Pinscher", "Greater Swiss Mountain Dog", "Bernese Mountain Dog",
                        "Appenzeller Sennenhund", "Entlebucher Sennenhund", "Boxer", "Bullmastiff", "Tibetan Mastiff",
                        "French Bulldog", "Great Dane", "St. Bernard", "husky", "Alaskan Malamute", "Siberian Husky",
                        "Dalmatian", "Affenpinscher", "Basenji", "pug", "Leonberger", "Newfoundland dog",
                        "Great Pyrenees dog", "Samoyed", "Pomeranian", "Chow Chow", "Keeshond", "brussels griffon",
                        "Pembroke Welsh Corgi", "Cardigan Welsh Corgi", "Toy Poodle", "Miniature Poodle",
                        "Standard Poodle", "Mexican hairless dog (xoloitzcuintli)", "grey wolf", "Alaskan tundra wolf",
                        "red wolf or maned wolf", "coyote", "dingo", "dhole", "African wild dog", "hyena", "red fox",
                        "kit fox", "Arctic fox", "grey fox", "tabby cat", "tiger cat", "Persian cat", "Siamese cat",
                        "Egyptian Mau", "cougar", "lynx", "leopard", "snow leopard", "jaguar", "lion", "tiger",
                        "cheetah", "brown bear", "American black bear", "polar bear", "sloth bear", "mongoose",
                        "meerkat", "tiger beetle", "ladybug", "ground beetle", "longhorn beetle", "leaf beetle",
                        "dung beetle", "rhinoceros beetle", "weevil", "fly", "bee", "ant", "grasshopper",
                        "cricket insect", "stick insect", "cockroach", "praying mantis", "cicada", "leafhopper",
                        "lacewing", "dragonfly", "damselfly", "red admiral butterfly", "ringlet butterfly",
                        "monarch butterfly", "small white butterfly", "sulphur butterfly", "gossamer-winged butterfly",
                        "starfish", "sea urchin", "sea cucumber", "cottontail rabbit", "hare", "Angora rabbit",
                        "hamster", "porcupine", "fox squirrel", "marmot", "beaver", "guinea pig", "common sorrel horse",
                        "zebra", "pig", "wild boar", "warthog", "hippopotamus", "ox", "water buffalo", "bison",
                        "ram (adult male sheep)", "bighorn sheep", "Alpine ibex", "hartebeest", "impala (antelope)",
                        "gazelle", "arabian camel", "llama", "weasel", "mink", "European polecat",
                        "black-footed ferret", "otter", "skunk", "badger", "armadillo", "three-toed sloth", "orangutan",
                        "gorilla", "chimpanzee", "gibbon", "siamang", "guenon", "patas monkey", "baboon", "macaque",
                        "langur", "black-and-white colobus", "proboscis monkey", "marmoset", "white-headed capuchin",
                        "howler monkey", "titi monkey", "Geoffroy's spider monkey", "common squirrel monkey",
                        "ring-tailed lemur", "indri", "Asian elephant", "African bush elephant", "red panda",
                        "giant panda", "snoek fish", "eel", "silver salmon", "rock beauty fish", "clownfish",
                        "sturgeon", "gar fish", "lionfish", "pufferfish", "abacus", "abaya", "academic gown",
                        "accordion", "acoustic guitar", "aircraft carrier", "airliner", "airship", "altar", "ambulance",
                        "amphibious vehicle", "analog clock", "apiary", "apron", "trash can", "assault rifle",
                        "backpack", "bakery", "balance beam", "balloon", "ballpoint pen", "Band-Aid", "banjo",
                        "baluster handrail", "barbell", "barber chair", "barbershop", "barn", "barometer", "barrel",
                        "wheelbarrow", "baseball", "basketball", "bassinet", "bassoon", "swimming cap", "bath towel",
                        "bathtub", "station wagon", "lighthouse", "beaker", "military hat (bearskin or shako)",
                        "beer bottle", "beer glass", "bell tower", "baby bib", "tandem bicycle", "bikini",
                        "ring binder", "binoculars", "birdhouse", "boathouse", "bobsleigh", "bolo tie", "poke bonnet",
                        "bookcase", "bookstore", "bottle cap", "hunting bow", "bow tie", "brass memorial plaque", "bra",
                        "breakwater", "breastplate", "broom", "bucket", "buckle", "bulletproof vest",
                        "high-speed train", "butcher shop", "taxicab", "cauldron", "candle", "cannon", "canoe",
                        "can opener", "cardigan", "car mirror", "carousel", "tool kit", "cardboard box carton",
                        "car wheel", "automated teller machine", "cassette", "cassette player", "castle", "catamaran",
                        "CD player", "cello", "mobile phone", "chain", "chain-link fence", "chain mail", "chainsaw",
                        "storage chest", "chiffonier", "bell or wind chime", "china cabinet", "Christmas stocking",
                        "church", "movie theater", "cleaver", "cliff dwelling", "cloak", "clogs", "cocktail shaker",
                        "coffee mug", "coffeemaker", "spiral or coil", "combination lock", "computer keyboard",
                        "candy store", "container ship", "convertible", "corkscrew", "cornet", "cowboy boot",
                        "cowboy hat", "cradle", "construction crane", "crash helmet", "crate", "infant bed",
                        "Crock Pot", "croquet ball", "crutch", "cuirass", "dam", "desk", "desktop computer",
                        "rotary dial telephone", "diaper", "digital clock", "digital watch", "dining table",
                        "dishcloth", "dishwasher", "disc brake", "dock", "dog sled", "dome", "doormat", "drilling rig",
                        "drum", "drumstick", "dumbbell", "Dutch oven", "electric fan", "electric guitar",
                        "electric locomotive", "entertainment center", "envelope", "espresso machine", "face powder",
                        "feather boa", "filing cabinet", "fireboat", "fire truck", "fire screen", "flagpole", "flute",
                        "folding chair", "football helmet", "forklift", "fountain", "fountain pen", "four-poster bed",
                        "freight car", "French horn", "frying pan", "fur coat", "garbage truck",
                        "gas mask or respirator", "gas pump", "goblet", "go-kart", "golf ball", "golf cart", "gondola",
                        "gong", "gown", "grand piano", "greenhouse", "radiator grille", "grocery store", "guillotine",
                        "hair clip", "hair spray", "half-track", "hammer", "hamper", "hair dryer", "hand-held computer",
                        "handkerchief", "hard disk drive", "harmonica", "harp", "combine harvester", "hatchet",
                        "holster", "home theater", "honeycomb", "hook", "hoop skirt", "gymnastic horizontal bar",
                        "horse-drawn vehicle", "hourglass", "iPod", "clothes iron", "carved pumpkin", "jeans", "jeep",
                        "T-shirt", "jigsaw puzzle", "rickshaw", "joystick", "kimono", "knee pad", "knot", "lab coat",
                        "ladle", "lampshade", "laptop computer", "lawn mower", "lens cap", "letter opener", "library",
                        "lifeboat", "lighter", "limousine", "ocean liner", "lipstick", "slip-on shoe", "lotion",
                        "music speaker", "loupe magnifying glass", "sawmill", "magnetic compass", "messenger bag",
                        "mailbox", "tights", "one-piece bathing suit", "manhole cover", "maraca", "marimba", "mask",
                        "matchstick", "maypole", "maze", "measuring cup", "medicine cabinet", "megalith", "microphone",
                        "microwave oven", "military uniform", "milk can", "minibus", "miniskirt", "minivan", "missile",
                        "mitten", "mixing bowl", "mobile home", "ford model t", "modem", "monastery", "monitor",
                        "moped", "mortar and pestle", "graduation cap", "mosque", "mosquito net", "vespa",
                        "mountain bike", "tent", "computer mouse", "mousetrap", "moving van", "muzzle", "metal nail",
                        "neck brace", "necklace", "baby pacifier", "notebook computer", "obelisk", "oboe", "ocarina",
                        "odometer", "oil filter", "pipe organ", "oscilloscope", "overskirt", "bullock cart",
                        "oxygen mask", "product packet packaging", "paddle", "paddle wheel", "padlock", "paintbrush",
                        "pajamas", "palace", "pan flute", "paper towel", "parachute", "parallel bars", "park bench",
                        "parking meter", "railroad car", "patio", "payphone", "pedestal", "pencil case",
                        "pencil sharpener", "perfume", "Petri dish", "photocopier", "plectrum", "Pickelhaube",
                        "picket fence", "pickup truck", "pier", "piggy bank", "pill bottle", "pillow", "ping-pong ball",
                        "pinwheel", "pirate ship", "drink pitcher", "block plane", "planetarium", "plastic bag",
                        "plate rack", "farm plow", "plunger", "Polaroid camera", "pole", "police van", "poncho",
                        "pool table", "soda bottle", "plant pot", "potter's wheel", "power drill", "prayer rug",
                        "printer", "prison", "projectile missile", "projector", "hockey puck", "punching bag", "purse", "quill",
                        "quilt", "race car", "racket", "radiator", "radio", "radio telescope", "rain barrel",
                        "recreational vehicle", "fishing casting reel", "reflex camera", "refrigerator",
                        "remote control", "restaurant", "revolver", "rifle", "rocking chair", "rotisserie", "eraser",
                        "rugby ball", "ruler measuring stick", "sneaker", "safe", "safety pin", "salt shaker", "sandal",
                        "sarong", "saxophone", "scabbard", "weighing scale", "school bus", "schooner", "scoreboard",
                        "CRT monitor", "screw", "screwdriver", "seat belt", "sewing machine", "shield", "shoe store",
                        "shoji screen room divider", "shopping basket", "shopping cart", "shovel", "shower cap",
                        "shower curtain", "ski", "balaclava ski mask", "sleeping bag", "slide rule", "sliding door",
                        "slot machine", "snorkel", "snowmobile", "snowplow", "soap dispenser", "soccer ball", "sock",
                        "solar thermal collector", "sombrero", "soup bowl", "keyboard space bar", "space heater",
                        "space shuttle", "spatula", "motorboat", "spider web", "spindle", "sports car", "spotlight",
                        "stage", "steam locomotive", "through arch bridge", "steel drum", "stethoscope", "scarf",
                        "stone wall", "stopwatch", "stove", "strainer", "tram", "stretcher", "couch", "stupa",
                        "submarine", "suit", "sundial", "sunglass", "sunglasses", "sunscreen", "suspension bridge",
                        "mop", "sweatshirt", "swim trunks shorts", "swing", "electrical switch", "syringe",
                        "table lamp", "tank", "tape player", "teapot", "teddy bear", "television", "tennis ball",
                        "thatched roof", "front curtain", "thimble", "threshing machine", "throne", "tile roof",
                        "toaster", "tobacco shop", "toilet seat", "torch", "totem pole", "tow truck", "toy store",
                        "tractor", "semi-trailer truck", "tray", "trench coat", "tricycle", "trimaran", "tripod",
                        "triumphal arch", "trolleybus", "trombone", "hot tub", "turnstile", "typewriter keyboard",
                        "umbrella", "unicycle", "upright piano", "vacuum cleaner", "vase", "vaulted or arched ceiling",
                        "velvet fabric", "vending machine", "vestment", "viaduct", "violin", "volleyball",
                        "waffle iron", "wall clock", "wallet", "wardrobe", "military aircraft", "sink",
                        "washing machine", "water bottle", "water jug", "water tower", "whiskey jug", "whistle",
                        "hair wig", "window screen", "window shade", "Windsor tie", "wine bottle", "airplane wing",
                        "wok", "wooden spoon", "wool", "split-rail fence", "shipwreck", "sailboat", "yurt", "website",
                        "comic book", "crossword", "traffic or street sign", "traffic light", "dust jacket", "menu",
                        "plate", "guacamole", "consomme", "hot pot", "trifle", "ice cream", "popsicle", "baguette",
                        "bagel", "pretzel", "cheeseburger", "hot dog", "mashed potatoes", "cabbage", "broccoli",
                        "cauliflower", "zucchini", "spaghetti squash", "acorn squash", "butternut squash", "cucumber",
                        "artichoke", "bell pepper", "cardoon", "mushroom", "Granny Smith apple", "strawberry", "orange",
                        "lemon", "fig", "pineapple", "banana", "jackfruit", "cherimoya (custard apple)", "pomegranate",
                        "hay", "carbonara", "chocolate syrup", "dough", "meatloaf", "pizza", "pot pie", "burrito",
                        "red wine", "espresso", "tea cup", "eggnog", "mountain", "bubble", "cliff", "coral reef",
                        "geyser", "lakeshore", "promontory", "sandbar", "beach", "valley", "volcano", "baseball player",
                        "bridegroom", "scuba diver", "rapeseed", "daisy", "yellow lady's slipper", "corn", "acorn",
                        "rose hip", "horse chestnut seed", "coral fungus", "agaric", "gyromitra", "stinkhorn mushroom",
                        "earth star fungus", "hen of the woods mushroom", "bolete", "corn cob", "toilet paper"]

# easy classified classes
# ['n02979186', 'n03888257', 'n03394916', 'n03445777', 'n03028079', 'n01440764', 'n02102040', 'n03000684', 'n03425413', 'n03417042']
imagenette = [0, 217, 482, 491, 497, 566, 569, 571, 574, 701]

# ["australian_terrier", "border_terrier", "samoyed", "beagle", "shih-tzu", "english_foxhound", "rhodesian_ridgeback", "dingo", "golden_retriever", "english_sheepdog"]
imagewoof = [193, 182, 258, 162, 155, 167, 159, 273, 207, 229]

# ["tabby_cat", "bengal_cat", "persian_cat", "siamese_cat", "egyptian_cat", "lion", "tiger", "jaguar", "snow_leopard", "lynx"]
imagemeow = [281, 282, 283, 284, 285, 291, 292, 290, 289, 287]

# ["peacock", "flamingo", "macaw", "pelican", "king_penguin", "bald_eagle", "toucan", "ostrich", "black_swan", "cockatoo"]
imagesquawk = [84, 130, 88, 144, 145, 22, 96, 9, 100, 89]

# ["pineapple", "banana", "strawberry", "orange", "lemon", "pomegranate", "fig", "bell_pepper", "cucumber", "green_apple"]
imagefruit = [953, 954, 949, 950, 951, 957, 952, 945, 943, 948]

# ["bee", "ladys slipper", "banana", "lemon", "corn", "school_bus", "honeycomb", "lion", "garden_spider", "goldfinch"]
imageyellow = [309, 986, 954, 951, 987, 779, 599, 291, 72, 11]

# 'n03787032', 'n02114855', 'n02877765', 'n02086910', 'n04099969', 'n03032252', 'n02119022', 'n03062245', 'n02116738', 'n02869837', 'n01980166', 'n04589890', 'n03777754', 'n02138441', 'n03379051', 'n02259212', 'n03492542', 'n03530642', 'n03785016', 'n02093428', 'n04429376', 'n04517823', 'n04067472', 'n02231487', 'n01978455', 'n04026417', 'n01983481', 'n02107142', 'n02105505', 'n02488291', 'n03584829', 'n04111531', 'n02859443', 'n03930630', 'n02091831', 'n07831146', 'n02804414', 'n02109047', 'n04336792', 'n04493381', 'n02018207', 'n02087046', 'n04435653', 'n03637318', 'n02123045', 'n03085013', 'n04485082', 'n02788148', 'n02100583', 'n02104029', 'n02108089', 'n13040303', 'n04592741', 'n01820546', 'n03903868', 'n02326432', 'n01729322', 'n07836838', 'n02085620', 'n01692333', 'n03837869', 'n03947888', 'n02701002', 'n02089973', 'n01749939', 'n02009229', 'n02483362', 'n02974003', 'n03891251', 'n02099849', 'n03794056', 'n03494278', 'n03424325', 'n04238763', 'n02106550', 'n03259280', 'n03017168', 'n01558993', 'n01773797', 'n04418357', 'n02113978', 'n07753275', 'n01735189', 'n02086240', 'n04127249', 'n01855672', 'n03775546', 'n04136333', 'n02089867', 'n02090622', 'n03642806', 'n02172182', 'n07714571', 'n07715103', 'n13037406', 'n02113799', 'n03764736', 'n02396427', 'n03594734', 'n04229816'
imagenet100 = [15, 45, 54, 57, 64, 74, 90, 99, 119, 120, 122, 131, 137, 151, 155, 157, 158, 166, 167, 169, 176, 180, 209, 211, 222, 228, 234, 236, 242, 246, 267, 268, 272, 275, 277, 281, 299, 305, 313, 317, 331, 342, 368, 374, 407, 421, 431, 449, 452, 455, 479, 494, 498, 503, 508, 544, 560, 570, 592, 593, 599, 606, 608, 619, 620, 653, 659, 662, 665, 667, 674, 682, 703, 708, 717, 724, 748, 758, 765, 766, 772, 775, 796, 798, 830, 854, 857, 858, 872, 876, 882, 904, 908, 936, 938, 953, 959, 960, 993, 994]

imageneta = [6, 11, 13, 15, 17, 22, 23, 27, 30, 37, 39, 42, 47, 50, 57, 70, 71, 76, 79, 89, 90, 94, 96, 97, 99, 105, 107, 108, 110, 113, 124, 125, 130, 132, 143, 144, 150, 151, 207, 234, 235, 254, 277, 283, 287, 291, 295, 298, 301, 306, 307, 308, 309, 310, 311, 313, 314, 315, 317, 319, 323, 324, 326, 327, 330, 334, 335, 336, 347, 361, 363, 372, 378, 386, 397, 400, 401, 402, 404, 407, 411, 416, 417, 420, 425, 428, 430, 437, 438, 445, 456, 457, 461, 462, 470, 472, 483, 486, 488, 492, 496, 514, 516, 528, 530, 539, 542, 543, 549, 552, 557, 561, 562, 569, 572, 573, 575, 579, 589, 606, 607, 609, 614, 626, 627, 640, 641, 642, 643, 658, 668, 677, 682, 684, 687, 701, 704, 719, 736, 746, 749, 752, 758, 763, 765, 768, 773, 774, 776, 779, 780, 786, 792, 797, 802, 803, 804, 813, 815, 820, 823, 831, 833, 835, 839, 845, 847, 850, 859, 862, 870, 879, 880, 888, 890, 897, 900, 907, 913, 924, 932, 933, 934, 937, 943, 945, 947, 951, 954, 956, 957, 959, 971, 972, 980, 981, 984, 986, 987, 988]

imagenetr = [1, 2, 4, 6, 8, 9, 11, 13, 22, 23, 26, 29, 31, 39, 47, 63, 71, 76, 79, 84, 90, 94, 96, 97, 99, 100, 105, 107, 113, 122, 125, 130, 132, 144, 145, 147, 148, 150, 151, 155, 160, 161, 162, 163, 171, 172, 178, 187, 195, 199, 203, 207, 208, 219, 231, 232, 234, 235, 242, 245, 247, 250, 251, 254, 259, 260, 263, 265, 267, 269, 276, 277, 281, 288, 289, 291, 292, 293, 296, 299, 301, 308, 309, 310, 311, 314, 315, 319, 323, 327, 330, 334, 335, 337, 338, 340, 341, 344, 347, 353, 355, 361, 362, 365, 366, 367, 368, 372, 388, 390, 393, 397, 401, 407, 413, 414, 425, 428, 430, 435, 437, 441, 447, 448, 457, 462, 463, 469, 470, 471, 472, 476, 483, 487, 515, 546, 555, 558, 570, 579, 583, 587, 593, 594, 596, 609, 613, 617, 621, 629, 637, 657, 658, 701, 717, 724, 763, 768, 774, 776, 779, 780, 787, 805, 812, 815, 820, 824, 833, 847, 852, 866, 875, 883, 889, 895, 907, 928, 931, 932, 933, 934, 936, 937, 943, 945, 947, 948, 949, 951, 953, 954, 957, 963, 965, 967, 980, 981, 983, 988]

imagenetvis = [0, 217]

imagenet1k = list(range(1000))

imagenet_subclass_dict = {
    "imagenette" : imagenette,
    "imagewoof" : imagewoof,
    "imagefruit": imagefruit,
    "imageyellow": imageyellow,
    "imagemeow": imagemeow,
    "imagesquawk": imagesquawk,
    "imagenet100": imagenet100,
    "imagenet1k": imagenet1k,
    "imagenetv2": imagenet1k,
    "imagenet_sketch":imagenet1k,
    "imageneta":imageneta,
    "imagenetr":imagenetr,
    "imagenetvis":imagenetvis,
}

ood_testset = ["imagenetv2","imagenet_sketch","imageneta","imagenetr"]

mean=(0.48145466, 0.4578275, 0.40821073)
std=(0.26862954, 0.26130258, 0.27577711)
train_preprocess = transforms.Compose([
                                            transforms.RandomResizedCrop(size=224, scale=(0.5, 1), interpolation=transforms.InterpolationMode.BICUBIC),
                                            transforms.RandomHorizontalFlip(p=0.5),
                                            transforms.ToTensor(),
                                            transforms.Normalize(mean=mean, std=std)
                                        ])
test_preprocess = transforms.Compose([
                                        transforms.Resize((224, 224)),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=mean, std=std)
                                        ])

def get_image_paths_from_file(file_path, subset):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    valid_labels = imagenet_subclass_dict[subset]
    
    image_paths, labels = [], []
    for line in lines:
        label = int(line.split()[-1])       
        if not label in valid_labels:
            continue
        if subset in ['imageneta']:
            image_paths.append(' '.join(line.split()[:-1]))

        else:
            image_paths.append(line.split()[0])
        labels.append(label)
    
    label_map = {value: index for index, value in enumerate(valid_labels)}
    
    print("Total Length:",len(image_paths),len(labels))    
    
    
    return image_paths, labels, label_map

class ImageNetCustom(Dataset):
    def __init__(self, root_dir="/zhaobai/ImageNet-1K/train", subset="imagenet1k", transform=None, file_list="file_list.txt"):
        self.root_dir = root_dir
        self.image_paths, self.labels, self.label_map = get_image_paths_from_file(os.path.join(self.root_dir,file_list), subset) 
        self.transform = transform
        self.subset = subset
        
    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_name = self.image_paths[idx]
        image_path = f"{self.root_dir}/{image_name}"
        label = self.labels[idx]
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        
        if self.subset not in ['imageneta','imagenetr']:
            
            label = self.label_map[label]
        
        return image, label

class ImageNetGeneration(Dataset):
    r"""
    Subset of a dataset at specified indices.

    Arguments:
        dataset (Dataset): The whole Dataset
        indices (sequence): Indices in the whole set selected for subset
        labels(sequence) : targets as required for the indices. will be the same length as indices
    """
    def __init__(self, root_dir="/zhaobai/ImageNet-1K/train", subset="imagenet1k", file_list="file_list.txt", use_caption=False):
        self.root_dir = root_dir
        self.image_paths, self.labels, self.label_map = get_image_paths_from_file(os.path.join(self.root_dir,file_list), subset) 
        self.caption_path = "/zhaobai46g/Datasets/ImageNet_BLIP2_caption_json/ImageNet_BLIP2_caption_json"
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_name = self.image_paths[idx]
        image_path = f"{self.root_dir}/{image_name}"
        label = self.labels[idx]
        # image = Image.open(image_path).convert("RGB")
        class_name = imagenet_classes[int(label)]
        image_name = image_name.split(".")[0][1:]
        
        return (label, image_path, image_name, class_name)

class CombinedImageNetCustom(Dataset):
    def __init__(self, root_dirs=["/zhaobai/ImageNet-1K/train"], subset="imagenet1k", transform=None, file_list="file_list.txt"):
        self.root_dirs = root_dirs
        self.transform = transform
        
        # Aggregate image paths and labels from all directories
        self.image_paths = []
        self.labels = []
        self.label_map = {}
        
        for root_dir in self.root_dirs:
            image_paths, labels, label_map = get_image_paths_from_file(os.path.join(root_dir, file_list), subset)
            
            # Include the full path (with root_dir) in self.image_paths
            full_image_paths = [f"{root_dir}/{image_path}" for image_path in image_paths]
            
            self.image_paths.extend(full_image_paths)
            self.labels.extend(labels)
            
            # Merge label maps (assuming they are consistent across datasets)
            self.label_map.update(label_map)
        
    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]  # Directly use the full path
        label = self.labels[idx]
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        
        label = self.label_map[label]
        
        return image, label

    
def get_dataset(root, split="train",subset='imagenet1k',filelist="file_list.txt",transform=None):
    if subset not in ood_testset:
        root_dir = os.path.join(root,split)
        preprocess = train_preprocess if split == 'train' else test_preprocess
    else:
        root_dir = root
        preprocess = test_preprocess
    
    if transform:
        preprocess = transform 
    dataset = ImageNetCustom(root_dir=root_dir, subset=subset, transform=preprocess, file_list=filelist)
    return dataset

def get_generation_dataset(root, split="train",subset='imagenet1k',filelist="file_list.txt"):
    root_dir = os.path.join(root,split)
    dataset = ImageNetGeneration(root_dir=root_dir, subset=subset, file_list=filelist)
    return dataset

def get_combined_dataset(roots,split="train",subset='imagenet1k',filelist="file_list.txt"):
    preprocess = train_preprocess if split == 'train' else test_preprocess
    dataset = CombinedImageNetCustom(root_dirs=roots, subset=subset, transform=preprocess, file_list=filelist)
    return dataset
    
        
        
if __name__ == "__main__":
    # root = "/zhaobai/ImageNet-1K/train"
    root = "/zhaobai/yuanjianhao/ImageNet_Syn_v124/train/"
    ds = ImageNetCustom(root_dir=root, subset="imagenette", transform=train_preprocess)
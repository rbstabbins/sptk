"""A store of material set definitions for the Spectral Parameters Toolkit

Defines the content of a material collection, following the template that
maps class labels to mineral group names and filenames or wildcards, e.g.:
materials = {'class_1': [('material_1', file_specification)],
            'class_2': [('material_2', file_specification),
                        ('material_3', file_specification)]}

Part of the Spectral Parameters Toolkit
Author: Roger Stabbins, NHM
Date: 31-08-2022
"""

# the test data set
TEST_SET = {'test_target': [('test_target', '*')],
            'test_background': [('test_background', '*')]}

# MacBeth ColorChecker Set
COLORCHECKER_SET = {
    'ColorChecker':[('naturals', 
                        ['dark_skin',
                        'light_skin',
                        'blue_sky',
                        'foliage',
                        'blue_flower',
                        'bluish_green']),
                    ('nonprimaries', 
                        ['orange',
                        'purplish_blue',
                        'moderate_red',
                        'purple',
                        'yellow_green',
                        'orange_yellow']),
                    ('primaries', 
                        ['blue',
                        'green',
                        'red',
                        'yellow',
                        'magenta',
                        'cyan']),
                    ('grayscales', 
                        ['black_2',
                        'neutral_3.5',
                        'neutral_5',
                        'neutral_6.5',
                        'neutral_8',
                        'white_9.5'])]
    }

# COLORCHECKER_SET = {
#     'natural': [('ColorChecker', 
#                  ['dark_skin',
#                   'light_skin',
#                   'blue_sky',
#                   'foliage',
#                   'blue_flower',
#                   'bluish_green'])],
#     'colour': [('ColorChecker', 
#                  ['orange',
#                   'purplish_blue',
#                   'moderate_red',
#                   'purple',
#                   'yellow_green',
#                   'orange_yellow'])],
#     'primary': [('ColorChecker', 
#                  ['blue',
#                   'green',
#                   'red',
#                   'yellow',
#                   'magenta',
#                   'cyan'])],
#     'grayscale': [('ColorChecker', 
#                  ['black',
#                   'neutral_3.5',
#                   'neutral_5',
#                   'neutral_6.5',
#                   'neutral_8',
#                   'white'])]
#     }

# hematite at oxia planum
OXIA_PLANUM_SET = {
    'hematite': [
                ('hematite', '*')],
    'basalt': [
                ('basalt', '*')],
    'clays':  [
                ('vermiculite', '*'),
                ('saponite', '*'),
                ('montmorillonite', '*')]}

# MICA files and categories
MICA_SET = {
    'iron oxides & primary silicates': [
            ('hematite', '*'),
            ('forsterite', '*'),
            ('fayalite', '*'),
            ('plagioclase', '*'),
            ('orthopyroxene', '*'),
            ('pyroxene', '*')],
    'ices': [
        ('h2o-ice', '*'),
        ('co2-ice', '*')
    ],
    'sulfates': [
        ('kieserite', '*'),
        ('alunite', '*'),
        ('jarosite', '*'),
        ('magnesium-sulfate', '*'),
        ('gypsum', '*'),
        ('bassanite', '*')
    ],
    'phyllosilicate': [
        ('montmorillonite', '*'),
        ('kaolinite', '*'),
        ('margarite', '*'),
        ('illite', '*'),
        ('nontronite', '*'),
        ('saponite', '*'),
        ('talc', '*'),
        ('serpentine', '*'),
        ('chlorite', '*')
    ],
    'carbonates': [
        ('magnesite', '*'),
        ('calcite', '*')
    ],
    'hydrated silicates & halides': [
        ('hydrated-silica', '*'),
        ('halite_mica', '*'),
        ('epidote', '*'),
        ('analcime', '*'),
        ('chloride', '*'),
        ('zeolite-prehnite', '*')
    ]
}

RELAB_CARBONATES = {'carbonate': [
            ('ankerite', '*'),
            ('aragonite', '*'),
            ('artinite', '*'),
            ('azurite', '*'),
            ('calcite', '*'),
            ('cerussite', '*'),
            ('dawsonite', '*'),
            ('dolomite', '*'),
            ('fe-carbonate', '*'),
            ('magnesite', '*'),
            ('malachite', '*'),
            ('manasseite', '*'),
            ('manganocalcite', '*'),
            ('mg-carbonate', '*'),
            ('monohydrocalcite', '*'),
            ('siderite', '*'),
            ('thermonatrite', '*'),
            ('trona', '*')
            ]}

RELAB_CYCLOSILICATES = {'cyclosilicate': [
        ('beryl', '*'),
        ('tourmaline', '*')
    ]}

RELAB_HALIDES = {'halide': [
        ('ammonium chloride', '*'),
        ('atacamite', '*'),
        ('calcium chloride', '*'),
        ('halite', '*'),
        ('iron (ii) chloride tetrahydrate', '*'),
        ('iron (iii) chloride', '*'),
        ('iron (iii) anhydrate', '*'),
        ('iron (iii) hexahydrate', '*'),
        ('magnesium chloride', '*'),
        ('magnesium chloride hexahydrate', '*'),
        ('paratacamite', '*'),
        ('sinjarite', '*')
    ]}


RELAB_HYDROXIDES = {'hydroxide': [
        ('akagenite', '*'),
        ('bohmite', '*'),
        ('brucite', '*'),
        ('diaspore', '*'),
        ('ferrihydrite', '*'),
        ('ferrihydrite alanine', '*'),
        ('ferrihydrite glucose', '*'),
        ('gibbsite', '*'),
        ('goethite', '*'),
        ('lepidocrocite', '*'),
        ('manganite', '*'),
        ('pyrochroite', '*')
    ]}

RELAB_INOSILICATES = {'inosilicate': [
        ('actinolite', '*'),
        ('allophane', '*'),
        ('amphibolite', '*'),
        ('anthophyllite', '*'),
        ('augite', '*'),
        ('clinopyroxene', '*'),
        ('diopside', '*'),
        ('enstatite', '*'),
        ('glaucophane', '*'),
        ('hedenbergite', '*'),
        ('hornblende', '*'),
        ('jeffersonite', '*'),
        ('orthopyroxene', '*'),
        ('pigeonite', '*'),
        ('pyroxene', '*'),
        ('riebeckite', '*'),
        ('tremolite', '*'),
        ('wollastonite', '*')
    ]}

RELAB_NESOSILICATES = {'nesosilicate': [
        ('andradite garnet', '*'),
        ('fayalite', '*'),
        ('forsterite', '*'),
        ('hortonolite', '*'),
        ('olivine', '*'),
        ('peridot', '*')
    ]}

RELAB_OXIDES = {'oxide': [
        ('anatase', '*'),
        ('brucite', '*'),
        ('chromite', '*'),
        ('corundum', '*'),
        ('ferric oxide', '*'),
        ('franklinite', '*'),
        ('gahnite', '*'),
        ('gibbsite', '*'),
        ('goethite', '*'),
        ('hematite', '*'),
        ('ilmenite', '*'),
        ('iron oxide', '*'),
        ('maghemite', '*'),
        ('magnetite', '*'),
        ('rutile', '*'),
        ('specular hematite', '*'),
        ('spinel', '*'),
        ('wustite', '*'),
        ('zincite', '*')
    ]}

RELAB_PHOSPHATES = {'phosphate': [
        ('alluadite', '*'),
        ('apatite', '*'),
        ('baricite', '*'),
        ('beraunite', '*'),
        ('chalcosiderite', '*'),
        ('childernite', '*'),
        ('fluorapatite', '*'),
        ('kidwellite', '*'),
        ('kulanite', '*'),
        ('strengite', '*'),
        ('strunzite', '*'),
        ('vivianite', '*'),
    ]}

RELAB_PHYLLOSILICATES = {'phyllosilicate': [
        ('allophane', '*'),
        ('annite', '*'),
        ('antigorite', '*'),
        ('attapulgite', '*'),
        ('beidellite', '*'),
        ('bentonite', '*'),
        ('berthierine', '*'),
        ('biotite', '*'),
        ('celadonite', '*'),
        ('chamosite', '*'),
        ('chewa nontronite', '*'),
        ('chlorite', '*'),
        ('chrysocolla', '*'),
        ('chrysotile', '*'),
        ('ferrosaponite', '*'),
        ('glauconite', '*'),
        ('greenalite', '*'),
        ('gyrolite', '*'),
        ('halloysite', '*'),
        ('hectorite', '*'),
        ('hissingerite', '*'),
        ('illite', '*'),
        ('imogonite', '*'),
        ('kaolin', '*'),
        ('kaolinite', '*'),
        ('lizardite', '*'),
        ('mica', '*'),
        ('montmorillonite', '*'),
        ('muscovite', '*'),
        ('neotocite', '*'),
        ('nontronite', '*'),
        ('palygorskite', '*'),
        ('phlogopite', '*'),
        ('pyrophyllite', '*'),
        ('ripidolite', '*'),
        ('saponite', '*'),
        ('sepiolite', '*'),
        ('serpentine', '*'),
        ('smectite', '*'),
        ('smectite fe-rich', '*'),
        ('smectite mg-rich', '*'),
        ('talc', '*'),
        ('tochilinite', '*'),
        ('vermiculite', '*')
    ]}

RELAB_SOROSILICATES = {'sorosilicate': [
        ('epidote', '*'),
        ('ilvaite', '*')
    ]}

RELAB_SULFATES = {'sulfate': [
        ('alunite', '*'),
        ('amarantite', '*'),
        ('anhydrite', '*'),
        ('aphthitalite', '*'),
        ('arcanite', '*'),
        ('barite', '*'),
        ('bilinite', '*'),
        ('botryogen', '*'),
        ('butlerite', '*'),
        ('celestine', '*'),
        ('copiapite', '*'),
        ('coquimbite', '*'),
        ('cornelite', '*'),
        ('ferrinatrite', '*'),
        ('fibroferrite', '*'),
        ('glauberite', '*'),
        ('goldichite', '*'),
        ('gypsum', '*'),
        ('hohmannite', '*'),
        ('hydrated iron sulfate', '*'),
        ('jarosite', '*'),
        ('kieserite', '*'),
        ('kornelite', '*'),
        ('lausenite', '*'),
        ('magnesium sulfate', '*'),
        ('lead sulfate', '*'),
        ('misenite', '*'),
        ('parabutlerite', '*'),
        ('paracoquimbite', '*'),
        ('pickeringite', '*'),
        ('rhomboclase', '*'),
        ('roemerite', '*'),
        ('rozenite', '*'),
        ('scapolite', '*'),
        ('schwertmannite', '*'),
        ('selenite', '*'),
        ('sideronatrite', '*'),
        ('slavikite', '*'),
        ('starkeyite', '*'),
        ('szomolnokite', '*'),
        ('thenardite', '*'),
        ('voltaite', '*'),
        ('yavapaiite', '*')
    ]}

RELAB_SULFIDES = {'sulfide': [
        ('chalcopyrite', '*'),
        ('galena', '*'),
        ('marcasite', '*'),
        ('mundrabilla troilite', '*'),
        ('paragould troilite', '*'),
        ('pentlandite', '*'),
        ('pyrite', '*'),
        ('sphalerite', '*'),
        ('stibnite', '*'),
        ('troilite', '*')
    ]}

RELAB_TECTOSILICATES = {'tectosilicate': [
        ('andesine', '*'),
        ('anorthite', '*'),
        ('anorthosite', '*'),
        ('buddingtonite', '*'),
        ('bytownite', '*'),
        ('cristobalite', '*'),
        ('feldspar', '*'),
        ('labradorite', '*'),
        ('mordenite', '*'),
        ('plagioclase', '*'),
        ('potassium feldspar', '*'),
        ('zeolite', '*'),
        ('zeolite thomsonite', '*'),
        ('zeolite stilbite', '*'),
        ('zeolite heulandite', '*'),
        ('zeolite barrerite', '*')
    ]}

RELAB_ALL_GROUPS = {
        'carbonate': RELAB_CARBONATES['carbonate'],
        'cyclosilicate': RELAB_CYCLOSILICATES['cyclosilicate'],
        'halide': RELAB_HALIDES['halide'],
        'hydroxide': RELAB_HYDROXIDES['hydroxide'],
        'inosilicate': RELAB_INOSILICATES['inosilicate'],
        'nesosilicate': RELAB_NESOSILICATES['nesosilicate'],
        'oxide': RELAB_OXIDES['oxide'],
        'phosphate': RELAB_PHOSPHATES['phosphate'],
        'phyllosilicate': RELAB_PHYLLOSILICATES['phyllosilicate'],
        'sorosilicate': RELAB_SOROSILICATES['sorosilicate'],
        'sulfide': RELAB_SULFIDES['sulfide'],
        'sulfate': RELAB_SULFATES['sulfate'],
        'tectosilicate': RELAB_TECTOSILICATES['tectosilicate']}

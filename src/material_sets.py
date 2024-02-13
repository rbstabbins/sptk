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
        ('halite', '*'),
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

RELAB_ALL_GROUPS = [
        RELAB_CARBONATES,
        RELAB_CYCLOSILICATES,
        RELAB_HALIDES,
        RELAB_HYDROXIDES,
        RELAB_INOSILICATES,
        RELAB_NESOSILICATES,
        RELAB_OXIDES,
        RELAB_PHOSPHATES,
        RELAB_PHYLLOSILICATES,
        RELAB_SOROSILICATES,
        RELAB_SULFIDES,
        RELAB_SULFATES,
        RELAB_TECTOSILICATES]

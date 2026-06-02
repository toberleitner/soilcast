from dataclasses import dataclass
from typing import Literal, Optional

cls_options = {
    1: 'Productive - cereal', 
    2: 'Productive - wide-row dominated',
    3: 'Productive - mixed',
    4: 'Integrated - cereal',
    5: 'Integrated - wide-row dominated',
    6: 'Integrated - legume dominated',
    7: 'Integrated - mixed'
}

till_options = {
    'conv': 'Conventional',
    'mintill': 'Min/no tillage'
}

irr_options = {
    'rf': 'Rainfed',
    'irr': 'Irrigated'
}

ssp_options = {
    '126': 'SSP 126',
    '585': 'SSP 585'
}
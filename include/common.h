const std::vector<std::string> CLASS_NAMES = {
    "back","lunzi","hole","plate_tray","plate_cell","plate_hole","pallet" };

enum CLASS_IDS {
    BACK = 0,LUNZI = 1,HOLE = 2,
    PLATE_TRAY = 3,PLATE_CELL = 4,PLATE_HOLE = 5,
    PALLET = 6
};

// feilun
const std::vector<std::string> CLASS_NAMES_FEILUN = {
    "back","boss","wheel"};

enum CLASS_IDS_FEILUN {
    BACK_FEILUN = 0,BOSS = 1, WHEEL= 2
};

// screen
const std::vector<std::string> CLASS_NAMES_SCREEN = {
    "background","pallet","circle"};

enum CLASS_IDS_SCREEN {
    BACKGROUND_SCREEN = 0,PALLET_SCREEN = 1,CIRCLE_SCREEN = 2
};

const std::vector<std::string> CLASS_NAMES_LUOWEN = {
    "back","no_screw","piece_screw",
    "yes_screw"};

enum CLASS_IDS_LUOWEN {
    BACK_LUOWEN = 0,no_screw = 1,piece_screw = 2,
    yes_screw = 3
};

const std::vector<std::vector<unsigned int>> COLORS = {
    {0, 114, 189},   {217, 83, 25},   {237, 177, 32},  {126, 47, 142},  {119, 172, 48},  {77, 190, 238},
    {162, 20, 47},   {76, 76, 76},    {153, 153, 153}, {255, 0, 0},     {255, 128, 0},   {191, 191, 0},
    {0, 255, 0},     {0, 0, 255},     {170, 0, 255},   {85, 85, 0},     {85, 170, 0},    {85, 255, 0},
    {170, 85, 0},    {170, 170, 0},   {170, 255, 0},   {255, 85, 0},    {255, 170, 0},   {255, 255, 0},
    {0, 85, 128},    {0, 170, 128},   {0, 255, 128},   {85, 0, 128},    {85, 85, 128},   {85, 170, 128},
    {85, 255, 128},  {170, 0, 128},   {170, 85, 128},  {170, 170, 128}, {170, 255, 128}, {255, 0, 128},
    {255, 85, 128},  {255, 170, 128}, {255, 255, 128}, {0, 85, 255},    {0, 170, 255},   {0, 255, 255},
    {85, 0, 255},    {85, 85, 255},   {85, 170, 255},  {85, 255, 255},  {170, 0, 255},   {170, 85, 255},
    {170, 170, 255}, {170, 255, 255}, {255, 0, 255},   {255, 85, 255},  {255, 170, 255}, {85, 0, 0},
    {128, 0, 0},     {170, 0, 0},     {212, 0, 0},     {255, 0, 0},     {0, 43, 0},      {0, 85, 0},
    {0, 128, 0},     {0, 170, 0},     {0, 212, 0},     {0, 255, 0},     {0, 0, 43},      {0, 0, 85},
    {0, 0, 128},     {0, 0, 170},     {0, 0, 212},     {0, 0, 255},     {0, 0, 0},       {36, 36, 36},
    {73, 73, 73},    {109, 109, 109}, {146, 146, 146}, {182, 182, 182}, {219, 219, 219}, {0, 114, 189},
    {80, 183, 189},  {128, 128, 0} };
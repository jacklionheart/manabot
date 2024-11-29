// basic_lands.h
#pragma once

#include <memory>

#include "rules/card.h"
// Function declarations
Card createBasicLandCard(const std::string& name, Color color);

Card basicPlains();
Card basicIsland();
Card basicMountain();
Card basicForest();
Card basicSwamp();

void registerBasicLands();
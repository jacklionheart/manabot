#pragma once

#include <vector>
#include <memory>
#include "rules/zones/zone.h"

class Game;
class Card;
class Player;

class StackObject {
public:
    virtual void resolve(Game* game) = 0;
    virtual ~StackObject() = default;
};

class Spell : public StackObject {
public:
    static int next_id;

    int id;

    Card* card;
    Player* controller;

    Spell(Card* card);

    void resolve(Game* game);
};

class Stack : public Zone {
public:
    std::vector<std::unique_ptr<Spell>> objects;

    Stack(Game* game);

    void cast(Card* card);
    void resolveTop();

    virtual void move(Card* card) override;
    virtual void remove(Card* card) override;

    size_t size() const;
};

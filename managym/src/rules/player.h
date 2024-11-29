#pragma once

#include <string>
#include <memory>
#include <map>
#include <vector>

class Game;
class Action;
class Permanent;
class Agent;
class Deck;

class PlayerConfig
{
public:
    std::string name;
    std::map<std::string, int> decklist;

    std::unique_ptr<Deck> instantiateDeck() const;

    PlayerConfig(const std::string &name, const std::map<std::string, int> &cardQuantities) : name(name), decklist(cardQuantities) {}
};

class Player
{
public:
    static int next_id;
    int id;

    std::unique_ptr<Agent> agent;
    std::unique_ptr<Deck> deck;

    std::string name;
    int life = 20;
    bool alive = true;

    Player(const PlayerConfig &config);

    void takeDamage(int damage);

    std::string toString() const;

private:
    std::unique_ptr<Deck> instantiateDeck(const PlayerConfig &config);
};

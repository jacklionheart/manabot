#pragma once 

#include <vector>
#include <map>
#include <memory>
#include "rules/zones/zone.h"

// Forward declarations
class Player;
class Mana;
class ManaCost;
class Game;

class Battlefield : public Zone {
public:
    std::map<Player*, std::vector<std::unique_ptr<Permanent>>> permanents;
    
    Battlefield(Game* game);

    virtual void move(Card* card) override;
    virtual void remove(Card* card) override;

    void destroy(Permanent* permanent);

    void enter(Card* card);

    void forEach(std::function<void(Permanent*)> func);
    void forEach(std::function<void(Permanent*)> func, Player* player);

    std::vector<Permanent*> attackers(Player* player);

    std::vector<Permanent*> eligibleAttackers(Player* player);
    std::vector<Permanent*> eligibleBlockers(Player* player);

    Permanent* find(const Card& card);

    Mana producibleMana(Player* player) const;
    void produceMana(const ManaCost& mana_cost, Player* player);
};

// Forward Declarations
class Mana;
class Player;
class Game;
class ManaAbility;
class ActivatedAbility;

class Permanent {
public:
    int id;
    static int next_id;

    // TODO: make an ID?
    Card* card;
    Player* controller;
    bool tapped = false;
    bool summoning_sick = false;
    int damage = 0;
    bool attacking = false;
    
    Permanent(Card* card);

    bool canTap() const;
    bool canAttack();
    bool canBlock();
    void untap();
    void tap();
    void takeDamage(int damage);
    bool hasLethalDamage() const;
    void clearDamage();
    void attack();
    Mana producibleMana(Game* game) const;
    void activateAllManaAbilities(Game* game);
    void activateAbility(ActivatedAbility* ability, Game* game);

    bool operator==(const Permanent& other) const;
};
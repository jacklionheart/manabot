#pragma once

#include <string>
#include <vector>
#include <set>
#include <memory>

#include "rules/mana.h"

class Game;
class Permanent;
class Zone;
class Player; 

class ActivatedAbility {
public:
    static int next_id;
    int id;
    bool uses_stack;

    ActivatedAbility();

    virtual void payCost(Permanent* permanent, Game* game) = 0;
    virtual bool canBeActivated(const Permanent* permanent, const Game* game) const = 0;
    virtual void resolve(Permanent* permanent, Game* game) = 0;
    virtual ~ActivatedAbility() = default;
};

class ManaAbility : public ActivatedAbility {
public:
    Mana mana;

    ManaAbility(const Mana& mana);

    void payCost(Permanent* permanent, Game* game) override;
    bool canBeActivated(const Permanent* permanent, const Game* game) const override;
    void resolve(Permanent* permanent, Game* game) override;
};

enum class CardType {
    CREATURE,
    INSTANT,
    SORCERY,
    PLANESWALKER,
    LAND,
    ENCHANTMENT,
    ARTIFACT,
    KINDRED,
    BATTLE
};

class CardTypes {
public:
    std::set<CardType> types;
    CardTypes() = default;
    CardTypes(const std::set<CardType>& types);
    bool isCastable() const;
    bool isPermanent() const;
    bool isNonLandPermanent() const;
    bool isNonCreaturePermanent() const;
    bool isSpell() const;
    bool isCreature() const;
    bool isLand() const;    
    bool isPlaneswalker() const;
    bool isEnchantment() const;
    bool isArtifact() const;
    bool isTribal() const;
    bool isBattle() const;
};

class Card {
public:
    static int next_id;
    int id;

    std::string name;
    std::optional<ManaCost> mana_cost;
    Colors colors;
    CardTypes types;
    std::vector<std::string> supertypes;
    std::vector<std::string> subtypes;
    std::vector<ManaAbility> mana_abilities;
    std::string text_box;
    std::optional<int> power;      // Use std::optional for optional values
    std::optional<int> toughness;
    Player* owner;
    Zone* current_zone;

    std::string toString() const;
    void removeFromCurrentZone();
    bool operator==(const Card* other) const;

    Card(const std::string& name,
         std::optional<ManaCost> mana_cost,
         const CardTypes& types,
         const std::vector<std::string>& supertypes,
         const std::vector<std::string>& subtypes,
         const std::vector<ManaAbility>& mana_abilities,
         const std::string& text_box,
         std::optional<int> power,
         std::optional<int> toughness);

    Card(const Card& other);

};

using Deck = std::vector<std::unique_ptr<Card>>;

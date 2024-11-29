// card.cpp
#include "card.h"
#include "rules/zones/zone.h"
#include "cardsets/card_registry.h"

#include "rules/game.h"

int ActivatedAbility::next_id = 0;

ActivatedAbility::ActivatedAbility() : id(next_id++) {
    uses_stack = true;
}

ManaAbility::ManaAbility(const Mana& mana) : ActivatedAbility(), mana(mana) {
    uses_stack = false;
}

void ManaAbility::payCost(Permanent* permanent, Game* game) {
    permanent->tap();
}

bool ManaAbility::canBeActivated(const Permanent* permanent, const Game* game) const {
    return permanent->canTap();
}

void ManaAbility::resolve(Permanent* permanent, Game* game) {
    game->addMana(permanent->controller, mana);
}

ManaAbility tapForMana(const std::string& mana_str) {
    return ManaAbility(Mana::parse(mana_str));
}

CardTypes::CardTypes(const std::set<CardType>& types) : types(types) {}

bool CardTypes::isPermanent() const {
    return types.count(CardType::CREATURE) ||
           types.count(CardType::LAND) ||
           types.count(CardType::ARTIFACT) ||
           types.count(CardType::ENCHANTMENT) ||
           types.count(CardType::PLANESWALKER) ||
           types.count(CardType::BATTLE);
}

bool CardTypes::isNonLandPermanent() const {
    return isPermanent() && !isLand();
}

bool CardTypes::isNonCreaturePermanent() const {
    return isPermanent() && !isCreature();
}

bool CardTypes::isCastable() const {
    return !isLand() && !types.empty();
}

bool CardTypes::isSpell() const {
    return types.contains(CardType::INSTANT) || types.contains(CardType::SORCERY);
}

bool CardTypes::isCreature() const {
    return types.contains(CardType::CREATURE);
}

bool CardTypes::isLand() const {
    return types.contains(CardType::LAND);
}

bool CardTypes::isPlaneswalker() const {
    return types.count(CardType::PLANESWALKER);
}

bool CardTypes::isEnchantment() const {
    return types.contains(CardType::ENCHANTMENT);
}

bool CardTypes::isArtifact() const {
    return types.contains(CardType::ARTIFACT);
}

bool CardTypes::isTribal() const {
    return types.contains(CardType::KINDRED);
}

bool CardTypes::isBattle() const {
    return types.contains(CardType::BATTLE);
}

int Card::next_id = 0;

Card::Card(const std::string& name,
           std::optional<ManaCost> mana_cost,
           const CardTypes& types,
           const std::vector<std::string>& supertypes,
           const std::vector<std::string>& subtypes,
           const std::vector<ManaAbility>& mana_abilities,
           const std::string& text_box,
           std::optional<int> power,
           std::optional<int> toughness)
    : 
      id(next_id++),
      name(name),
      mana_cost(mana_cost),
      types(types),
      supertypes(supertypes),
      subtypes(subtypes),
      mana_abilities(mana_abilities),
      text_box(text_box),
      power(power),
      toughness(toughness),
      owner(nullptr), // to be set by game
      current_zone(nullptr) {

    if (mana_cost.has_value()) {
        colors = mana_cost->colors();
    } else {
        colors = Colors();
    }
}

Card::Card(const Card& other)
    : id(next_id++),  // Assign a new unique ID
      name(other.name),
      mana_cost(other.mana_cost),
      colors(other.colors),
      types(other.types),
      supertypes(other.supertypes),
      subtypes(other.subtypes),
      mana_abilities(other.mana_abilities),
      text_box(other.text_box),
      power(other.power),
      toughness(other.toughness),
      owner(nullptr),           // Will be set when the game assigns ownership
      current_zone(nullptr) {}

void Card::removeFromCurrentZone() {
    if (current_zone) {
        current_zone->remove(this);
    }
    current_zone = nullptr;
}

bool Card::operator==(const Card* other) const {
    return this->id == other->id;
}

std::string Card::toString() const {
    return "{name: " + name + "}";
}


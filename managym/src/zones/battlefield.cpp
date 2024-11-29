// battlefield.cpp
#include "battlefield.h"
#include "rules/game.h"

#include <cassert>
#include <spdlog/spdlog.h>


Battlefield::Battlefield(Game* game)
    : Zone(game) {
    // Initialize the permanents map for each player
    for (std::unique_ptr<Player>& player : game->players) {
        cards[player.get()] = std::vector<Card*>();
        permanents[player.get()] = std::vector<std::unique_ptr<Permanent>>();
    }
}

void Battlefield::move(Card* card) {
    enter(card);     
}


void Battlefield::enter(Card* card) {
    if (!card->types.isPermanent()) {
        throw std::invalid_argument("Card is not a permanent: " + card->toString());
    }
    spdlog::info("{} enters battlefield", card->toString());
    Zone::move(card);
    Player* controller = card->owner;
    permanents[controller].push_back(std::make_unique<Permanent>(card));
}

void Battlefield::remove(Card* card) {
    Zone::remove(card);
    Player* controller = card->owner;
    permanents[controller].erase(
        std::remove_if(permanents[controller].begin(), permanents[controller].end(),
            [&card](const std::unique_ptr<Permanent>& permanent) { return &permanent->card == card; }),
        permanents[controller].end());
}

void Battlefield::destroy(Permanent* permanent) {
    game->zones->graveyard->move(permanent->card);

    spdlog::info("{} is destroyed", permanent->card->toString());
    Player* controller = permanent->card->owner;
    permanents[controller].erase(
        std::remove_if(permanents[controller].begin(), permanents[controller].end(),
            [&permanent](const std::unique_ptr<Permanent>& p) { return p.get() == permanent; }),
        permanents[controller].end());
}

Permanent* Battlefield::find(const Card* card) {
    for (const auto& [player, player_permanents] : permanents) {
        for (const std::unique_ptr<Permanent>& permanent : player_permanents) {
            if (permanent->card == card) {
                return permanent.get();
            }
        }
    }
    return nullptr;
}
void Battlefield::forEach(std::function<void(Permanent*)> func) {
    for (const auto& [player, player_permanents] : permanents) {
        for (const std::unique_ptr<Permanent>& permanent : player_permanents) {
            func(permanent.get());
        }
    }
}

void Battlefield::forEach(std::function<void(Permanent*)> func, Player* player) {
    std::vector<std::unique_ptr<Permanent>>& player_permanents = permanents[player];
    for (const std::unique_ptr<Permanent>& permanent : player_permanents) {
        func(permanent.get());
    }
}

std::vector<Permanent*> Battlefield::attackers(Player* player) {
    std::vector<Permanent*> attackers;
    std::vector<std::unique_ptr<Permanent>>& player_permanents = permanents[player];
    for (const std::unique_ptr<Permanent>& permanent : player_permanents) {
        if (permanent->attacking) {
            attackers.push_back(permanent.get());
        }
    }
    return attackers;
}

std::vector<Permanent*> Battlefield::eligibleAttackers(Player* player) {
    std::vector<Permanent*> eligible_attackers;
    
    std::vector<std::unique_ptr<Permanent>>& player_permanents = permanents[player];
    for (const std::unique_ptr<Permanent>& permanent : player_permanents) {
        if (permanent->canAttack()) {
            eligible_attackers.push_back(permanent.get());
        }
    }
    return eligible_attackers;
}   

std::vector<Permanent*> Battlefield::eligibleBlockers(Player* player) {
    std::vector<Permanent*> eligible_blockers;    
    
    std::vector<std::unique_ptr<Permanent>>& player_permanents = permanents[player];
    for (const std::unique_ptr<Permanent>& permanent : player_permanents) {
        if (permanent->canBlock()) {
            eligible_blockers.push_back(permanent.get());
        }
    }
    return eligible_blockers;
}   

Mana Battlefield::producibleMana(Player* player) const {
    Mana total_mana;
    auto it = permanents.find(player);
    if (it != permanents.end()) {
        for (const auto& permanent : it->second) {
            total_mana.add(permanent->producibleMana(game));
        }
    }
    return total_mana;
}

void Battlefield::produceMana(const ManaCost& mana_cost, Player* player) {
    Mana producible = producibleMana(player);
    if (!producible.canPay(mana_cost)) {
        throw std::runtime_error("Not enough producible mana to pay for mana cost.");
    }

    std::vector<std::unique_ptr<Permanent>>& player_permanents = permanents[player];
    for (const std::unique_ptr<Permanent>& permanent : player_permanents) {
        if (!game->mana_pools[player].canPay(mana_cost)) {
            permanent->activateAllManaAbilities(game);
        } else {
            break;
        }
    }

    if (!game->mana_pools[player].canPay(mana_cost)) {
        throw std::runtime_error("Did not generate enough mana to pay for mana cost.");
    }
}

int Permanent::next_id = 0;

Permanent::Permanent(Card* card) :
      controller(card->owner),
      id(next_id++),
      card(card) {

    tapped = false;
    summoning_sick = card->types.isCreature();
    damage = 0;
}

bool Permanent::canTap() const {
    return !tapped && !(summoning_sick && card->types.isCreature());
}

void Permanent::untap() {
    tapped = false;
}
void Permanent::tap() {
    spdlog::info("Tapping {}", card->toString());
    tapped = true;
}

void Permanent::takeDamage(int dmg) {
    damage += dmg;
}

bool Permanent::hasLethalDamage() const {
    return card->types.isCreature() && damage >= card->toughness;
}

bool Permanent::canAttack() {
    return card->types.isCreature() && !tapped && !summoning_sick;
}

bool Permanent::canBlock() {
    return card->types.isCreature() && !tapped;
}

void Permanent::clearDamage() {
    damage = 0;
}

void Permanent::attack() {
    attacking = true;
    tap();
}
Mana Permanent::producibleMana(Game* game) const {
    Mana totalMana;
    for (const ManaAbility& ability : card->mana_abilities) {
        if (ability.canBeActivated(this, game)) {
            totalMana.add(ability.mana);
        }
    }
    return totalMana;
}

void Permanent::activateAllManaAbilities(Game* game) {
    for (ManaAbility& ability : card->mana_abilities) {
        if (ability.canBeActivated(this, game)) {
            activateAbility(&ability, game);
        }
    }
}

void Permanent::activateAbility(ActivatedAbility* ability, Game* game) {
    spdlog::info("Activating ability on {}", card->toString());
    assert(ability != nullptr);
    if (!ability->canBeActivated(this, game)) {
        throw std::logic_error("Ability cannot be activated.");
    }
    ability->payCost(this, game);
    if (ability->uses_stack) {
        throw std::runtime_error("Implement me.");
        // TODO: Add to stack
        //game->addToStack(ability);
    } else {
        ability->resolve(this, game);
    }
}

bool Permanent::operator==(const Permanent& other) const {
    return this->id == other.id;
}

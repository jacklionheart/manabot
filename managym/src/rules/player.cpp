// player.cpp
#include "player.h"

#include "rules/game.h"
#include "cardsets/card_registry.h"

int Player::next_id = 0;
Player::Player(const PlayerConfig& config) : id(next_id++), name(config.name), deck(instantiateDeck(config)), agent(std::make_unique<Agent>(this)) {}

void Player::takeDamage(int damage) {
    life -= damage;
}

std::string Player::toString() const {
    return name + " (Player " + std::to_string(id) + " - Life: " + std::to_string(life) + ")";
}

std::unique_ptr<Deck> Player::instantiateDeck(const PlayerConfig& config) {
    CardRegistry& registry = CardRegistry::instance();
    std::unique_ptr<Deck> deck = std::make_unique<Deck>();

    for (const auto& [name, quantity] : config.decklist) {
        for (int i = 0; i < quantity; ++i) {
            std::unique_ptr<Card> card = registry.instantiate(name);
            card->owner = this;
            deck->push_back(std::move(card));
        }
    }
    return deck;
}
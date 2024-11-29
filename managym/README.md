# managym

managym is a C++ environment for reinforcement learning in Magic: The Gathering.

## Get Started

## Style 

### Casing

* Classes are PascalCase
* Variables are snake_case
* Methods are camelCase

### Objects and References

Basic objects are stored as member variables (i.e. Foo foo).  What is a basic objects?

 * Built-In Types (e.g. int, std::string, std::vector, etc.)
 * Simple types (e.g. Mana, ManaCost, Decklist etc.) that contain no references to other objects and do not use polymorphism

#### Object Pointers

Any object that contains pointers to other objects is itself stored as a pointer (e.g. *Bar bar) and not stored as members of the containing object. Every such object has exactly one owner using a unique_ptr. We do not use shared_ptr.


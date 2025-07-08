#pragma once

namespace madBasketball {

enum class CellFlag : uint32_t {
    Nothing = 0,
    Wall    = 1 << 0,
    End     = 1 << 1,
};

struct Cell {
    float reward;
    CellFlag flags;
};

struct GridState {
    const Cell *cells;

    float startX;
    float startY;
    float width;  // Width in meters
    float height; // Height in meters
    
    // Grid resolution for discrete cell lookups (cells per meter)
    int32_t cellsPerMeter;
    int32_t discreteWidth;  // Width in discrete cells
    int32_t discreteHeight; // Height in discrete cells
};

inline CellFlag & operator|=(CellFlag &a, CellFlag b)
{
    a = CellFlag(static_cast<uint32_t>(a) | static_cast<uint32_t>(b));
    return a;
}

inline bool operator&(CellFlag a, CellFlag b)
{
    return (static_cast<uint32_t>(a) & static_cast<uint32_t>(b)) > 0;
}

inline CellFlag operator|(CellFlag a, CellFlag b)
{
    a |= b;

    return a;
}

}

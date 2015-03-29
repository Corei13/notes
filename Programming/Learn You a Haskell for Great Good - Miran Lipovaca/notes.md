---
layout: page
title: Learn You a Haskell for Great Good
author: Miran Lipovaca
category: Programming
tags: Haskell
---

- **Chapter 1 - Introduction**
- **Chapter 2 - Starting Out**
    - *2.1 - Ready, Set, Go!*
        - invalid: `5 * -3`, valid `5 * (-3)`
        - Function application has the highest precedence
            `succ 9 + max 5 4 + 1` is equivalent to `(succ 9) + (max 5 4) + 1`
        - If a function takes two parameters, we can also call it as an infix function by surrounding it with backticks
            `div 92 10` is equivalent to `92 `div` 10`
    - *2.2 - Baby's First Functions*
        - `doubleMe x = x + x`
        - if must have an else

            ```haskell
            doubleSmallNumber x = if x > 100 then x else x * 2
            ```
        - function names can't start with uppercase
        - function can have arity zero

            ```haskell
            conanO'Brien = "It's a-me, Conan O'Brien!"
            ```
        - *2.3 - An Intro To Lists*
            - list must contain elements of same type
            - 'let' keyword is used to define names in GHCI

                ```haskell
                > let lostNumbers = [4,8,15,16,23,42]
                > lostNumbers
                [4,8,15,16,23,42]
                ```
            - single quote means character, double quotes means string
            - string is just a list of characters
            - lists can be contenated by `++` operator, complexity is linear

                ```haskell
                > [1,2,3,4] ++ [9,10,11,12]
                [1,2,3,4,9,10,11,12]
                > "hello" ++ " " ++ "world"
                "hello world"
                > ['w','o'] ++ ['o','t']
                "woot"
                ```
            - `:` operator (or *cons* operator) adds an element at the begining of a lost (`O(1)`)

                ```haskell
                > 'A':" SMALL CAT"
                "A SMALL CAT"
                > 5:[1,2,3,4,5]
                [5,1,2,3,4,5]
                ```
            - `[1, 2, 3]` is equivalent to `1:2:3:[]`
            - `!!` operator is used to access list elements, `[9.4,33.2,96.2,11.2,23.25] !! 1` is 33.2
            - lists are compared in lexicographical order
            - some list operations

                ```haskell
                head [5,4,3,2,1] -- 5
                tail [5,4,3,2,1] -- [4,3,2,1]
                last [5,4,3,2,1] -- 1
                init [5,4,3,2,1] -- [5,4,3,2]
                length [5,4,3,2,1] -- 5
                null [1,2,3] -- False
                null [] -- True
                reverse [5,4,3,2,1] -- [1,2,3,4,5]
                take 3 [5,4,3,2,1] -- [5,4,3]
                take 5 [1,2] -- [1,2]
                drop 3 [8,4,2,1,5,6] -- [1,5,6]
                drop 100 [1,2,3,4] -- []
                minimum [8,4,2,1,5,6] -- 1
                maximum [1,9,2,3,4] -- 9
                sum [5,2,1,6,3,2,5,7] -- 31
                product [6,2,1,2] -- 24
                elem 5 [1,3,5,7] -- True
                elem 4 [1,3,5,7] -- False
                ```
        - *2.4 - Texas Ranges*
            - range example

                ```haskell
                ['K'..'Z'] -- "KLMNOPQRSTUVWXYZ"
                [3,6..20] -- [3,6,9,12,15,18]
                [5,4..1] -- [5,4,3,2,1]
                [13,26..] -- [13,26,39,52,...]
                ```
            - new functions

                ```haskell
                cycle [1,2,3] -- [1,2,3,1,2,3,...]
                repeat 5 -- [5,5,5,...]
                replicate 3 10 -- [10,10,10]
                ```




















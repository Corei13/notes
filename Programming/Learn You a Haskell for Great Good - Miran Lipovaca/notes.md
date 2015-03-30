---
layout: page
title: Learn You a Haskell for Great Good
author: Miran Lipovaca
category: Programming
tags: Haskell
---

- **Chapter 1 - Starting Out**
    - *1.1 - Ready, Set, Go!*
        - Invalid: `5 * -3`, valid `5 * (-3)`
        - Function application has the highest precedence
            `succ 9 + max 5 4 + 1` is equivalent to `(succ 9) + (max 5 4) + 1`
        - If a function takes two parameters, we can also call it as an infix function by surrounding it with backticks
            `div 92 10` is equivalent to ``92 `div` 10``
    - *1.2 - Baby's First Functions*
        - If must have an else

            ```haskell
            doubleSmallNumber x = if x > 100 then x else x * 2
            ```
        - Function names can't start with uppercase, because uppercase means a type
        - Function can have arity zero

            ```haskell
            conanO'Brien = "It's a-me, Conan O'Brien!"
            ```
    - *1.3 - An Intro To Lists*
        - List must contain elements of same type
        - 'let' keyword is used to define names in GHCI

            ```haskell
            let lostNumbers = [4,8,15,16,23,42]
            lostNumbers -- [4,8,15,16,23,42]
            ```
        - Single quote means character, double quotes means string
        - String is just a list of characters
        - Lists can be contenated by `++` operator, complexity is linear

            ```haskell
            [1,2,3,4] ++ [9,10,11,12] -- [1,2,3,4,9,10,11,12]
            "hello" ++ " " ++ "world" -- "hello world"
            ['w','o'] ++ ['o','t'] -- "woot"
            ```
        - `:` operator (or *cons* operator) adds an element at the beginning of a lost (`O(1)`)

            ```haskell
            'A':" SMALL CAT" -- "A SMALL CAT"
            5:[1,2,3,4,5] -- [5,1,2,3,4,5]
            ```
        - `[1, 2, 3]` is equivalent to `1:2:3:[]`
        - `!!` operator is used to access list elements

            ```haskell
            [9.4,33.2,96.2,11.2,23.25] !! 1` -- 33.2
            ```
        - Lists are compared in lexicographical order
        - Some list operations

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
    - *1.4 - Texas Ranges*
        - Range example

            ```haskell
            ['K'..'Z'] -- "KLMNOPQRSTUVWXYZ"
            [3,6..20] -- [3,6,9,12,15,18]
            [5,4..1] -- [5,4,3,2,1]
            [13,26..] -- [13,26,39,52,...]
            ```
        - New functions

            ```haskell
            cycle [1,2,3] -- [1,2,3,1,2,3,...]
            repeat 5 -- [5,5,5,...]
            replicate 3 10 -- [10,10,10]
            ```
    - *1.5 - I'm A List Comprehension*
        - List comprehension examples

            ```haskell
            [x*2 | x <- [1..10]] -- [2,4,6,8,10,12,14,16,18,20]
            [x | x <- [50..100], x `mod` 7 == 3] -- [52,59,66,73,80,87,94]
            [x*y | x <- [2,5,10], y <- [8,10,11], x*y > 50] -- [55,80,100,110]
            length' xs = sum [1 | _ <- xs]
            take 5 [(i,j) | i <- [1..], j <- [1..i-1], gcd i j == 1] -- [(2,1),(3,1),(3,2),(4,1),(4,3)]
            take 5 [ (i,j) | i <- [1..], let k = i*i, j <- [1..k]] -- [(1,1),(2,1),(2,2),(2,3),(2,4)]
            ```
    - *1.6 - Tuples*
        - Can have different types of elements
        - Has its own type depending on its size and type of its elements
        - New functions

            ```haskell
            fst ("Wow", False) -- "Wow", works only on pair
            snd (8,11) -- 11, works only on pair
            zip [5,3,2,6,2,7,2,5,4,6,6] ["im","a","turtle"] -- [(5,"im"),(3,"a"),(2,"turtle")]
            ```
- **Chapter 2 - Believe The Type**
    - *2.1 - Explicit Type Declaration*
    - *2.2 - Common Haskell Types*
        - Some basic types: Bool, Char, Int, Float, Double, Integer (arbitrary precision integer)
        - Declaring functions with type

            ```haskell
            addThree :: Int -> Int -> Int -> Int
            addThree x y z = x + y + z
            factorial :: Integer -> Integer
            factorial n = product [1..n]
            factorial 50 -- 30414093201713378043612608166064768844377641568960512000000000000
            ```
    - *2.3 - Type Variables*
    - *2.4 - Typeclasses 101*
        - If a function is comprised only of special char- acters, it's considered an infix function by default. If we want to examine its type, pass it to another function or call it as a prefix function, we have to surround it in parentheses.

            ```haskell
            (==) 8 8 -- True
            (+) 7 8 -- 15
            ```
        - `Eq` typeclass provides an interface for types that can be tested for equality
        - `Ord` typeclass provides an interface for types that can be ordered
        - The `compare` function takes two `Ord` members of the same type and returns an ordering. Ordering is a type that can be `GT`, `LT` or `EQ`, meaning greater than, lesser than and equal, respectively.

            ```haskell
            "Abrakadabra" `compare` "Zebra" -- LT
            5 `compare` 3 -- GT
            ```
        - `show` transforms members of `Show` typeclass to `String`
        - `read` takes a string and returns a type of 'Read' typeclass
        - `X::Type` can be used to transform a typeclass `X` to `Type` whre `Type` is of typeclass `X`

            ```haskell
            show 5.334 -- "5.334"
            read "[1,2,3,4]" :: [Int] -- [1,2,3,4]
            read "(3, 'a')" :: (Float, Char) -- (3.0, 'a')
            minBound :: Int -- -2147483648
            maxBound :: Char -- '\1114111'
            maxBound :: (Bool, Int, Char) -- (True,2147483647,'\1114111')
            ```
        - Some other typeclasses: `Integral`, `Floating
- **Chapter 3 - Syntax In Functions**
    - *3.1 - Pattern matching*
        - Examples

            ```haskell
            addVectors :: (Num a) => (a, a) -> (a, a) -> (a, a)
            addVectors (x1, y1) (x2, y2) = (x1 + x2, y1 + y2)

            first :: (a, b, c) -> a
            first (x, _, _) = x

            second :: (a, b, c) -> b
            second (_, y, _) = y

            third :: (a, b, c) -> c
            third (_, _, z) = z


            let xs = [(1,3), (4,3), (2,4), (5,3), (5,6), (3,1)]
            [a+b | (a,b) <- xs] -- [4,7,6,8,11,4]

            head' :: [a] -> a
            head' [] = error "Can't call head on an empty list, dummy!"
            head' (x:_) = x
            ```
        - *as-patterns*: `@` can be used to refer patterns to use later

            ```haskell
            capital :: String -> String
            capital "" = "Empty string, whoops!"
            capital all@(x:xs) = "The first letter of " ++ all ++ " is " ++ [x]
            capital "Dracula" -- "The first letter of Dracula is D"
            ```
    - *3.2 - Guards, Guards!*
        - A *guard* is indicated by a pipe character (|), followed by a Boolean expression, followed by the function body that will be used if that expression evaluates to `True`. If the expression evaluates to `False`, the function drops through to the next guard. **Guards must be indented.**

            ```haskell
            max' :: (Ord a) => a -> a -> a
            max' a b
                | a <= b    = b
                | otherwise = a
            ```
    - *3.3 - Where?!*
        - Examples

            ```haskell
            bmiTell :: Double -> Double -> String
            bmiTell weight height
                | bmi <= skinny = "You're underweight, you emo, you!"
                | bmi <= normal = "You're supposedly normal. Pffft, I bet you're ugly!"
                | bmi <= fat    = "You're fat! Lose some weight, fatty!"
                | otherwise     = "You're a whale, congratulations!"
                where bmi = weight / height ^ 2 -- all variable names must be aligned in a single column
                      (skinny, normal, fat) = (18.5, 25.0, 30.0)

            initials :: String -> String -> String
            initials firstname lastname = [f] ++ ". " ++ [l] ++ "."
                where (f:_) = firstname
                      (l:_) = lastname

            calcBmis :: [(Double, Double)] -> [Double]
            calcBmis xs = [bmi w h | (w, h) <- xs]
                where bmi weight height = weight / height ^ 2
            ```
    - *3.4 - let It Be*
        - `let <bindings> in <expression>`
            
            ```haskell
            cylinder :: Double -> Double -> Double
            cylinder r h =
                let sideArea = 2 * pi * r * h
                    topArea = pi * r ^ 2
                in sideArea + 2 * topArea
            ```

        - `let` is an expression, so it has a value
        - `let` can be separated with semicolons in the same line

            ```haskell
            4 * (let a = 9 in a + 1) + 2 -- 42
            let square x = x * x in (square 5, square 3, square 2) -- (25, 9, 4)

            (let (a, b, c) = (1, 2, 3) in a*b*c, let foo="Hey "; bar = "there!" in foo ++ bar) -- (6,"Hey there!")
            ```
        - 'let' in list comprehension

            
            ```haskell
            calcBmis :: [(Double, Double)] -> [Double]
            calcBmis xs = [bmi | (w, h) <- xs, let bmi = w / h ^ 2, bmi > 25.0]
            ```
            The `(w, h) <- xs` is called *generator*. `bmi` can't be refered in the generator, because that is defined prior to the `let` binding.
        - `case` expression

            ```
            case <expression> of pattern -> result
                                 pattern -> result
                                 pattern -> result
                                 ...
            ```
            Example:

            ```haskell
            head' :: [a] -> a
            head' xs = case xs of [] -> error "No head for empty lists!"
                                  (x:_) -> x

            describeList :: [a] -> String
            describeList ls = "The list is " ++ case ls of [] -> "empty."
                                                           [x] -> "a singleton list."
                                                           xs -> "a longer list."
            ```
- **Chapter 4 - Hello Recursion!**
    - *4.1 - Maximum Awesome*
    - *4.2 - A Few More Recursive Functions*
    - *4.3 - Quick, Sort!*

        ```haskell
        quicksort :: (Ord a) => [a] -> [a]
        quicksort [] = []
        quicksort x:xs = quicksort left ++ [x] ++ quicksort right
            where left = [a | a <- xs, a <= x]
                  right = [a | a <- xs, a > x]
        ```
- **Chapter 5 - Higher-Order Functions**
    - *5.1 - Curried Functions*
        - Infix functions can be curried by *sections*. ``a `func` b`` is equivalent to ``(`func` b) a``
        - Be careful about negetive numbers. `(-4)` should have been ``(subtract 4)``, instead it is just `-4`
    - *5.2 - Some Higher-Orderism Is in Order*
        - Implementing flip

            ```haskell
            flip' :: (a -> b -> c) -> (b -> a -> c)
            flip' f = g
                where g x y = f y x
            -- another version
            flip' f y x = f x y -- this works because flip' f x y also defines flip' f x and flip' f
            ```
    - *5.3 - The Functional Programmer’s Toolbox*
        - `takeWhile :: (a -> Bool) -> [a] -> [a]` returns begining of a list until `f` is true

            ```haskell
            takeWhile (< 5) [1..] -- [1, 2, 3, 4]
            ```
    - *5.4 - Lambdas*
        - Examples

            ```haskell
            map (\(a,b) -> a + b) [(1,2),(3,5),(6,3),(2,6),(2,5)] -- [3,8,9,8,7]

            flip' :: (a -> b -> c) -> b -> a -> c
            flip' f = \x y -> f y x -- \x y -> (f x y)


            addThree :: Int -> Int -> Int -> Int
            addThree' = \x -> \y -> \z -> x + y + z -- (\x -> (\y -> (\z -> x + y + z)))
            ```


















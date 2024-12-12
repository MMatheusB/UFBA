barris = [
            0,
            6518,
            24144,
            21364,
            18076,
            14772,
            12225,
            10082,
            8338,
            6914,
            6100,
            5436,
            4786,
            4314,
            3928,
            3606,
            3331,
            3093,
            2887,
            2935,
        ]
soma = 0
for i in range (len(barris)):
    soma+= barris[i]

media = soma/len(barris)

print(media)
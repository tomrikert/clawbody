# Reachy emotion map

ClawBody exposes the 81 expressive recordings from Pollen's Reachy Mini
emotions dataset. Exact numbered names always select that recording; a family
name or conversational alias selects the first (restrained default) variant.

The four additional dataset recordings are utilities, not emotions:
`mini-deep-sleep`, `toc-toc-toc`, `waiting`, and `wake-mini-up`.

## Positive and social

- amazed: `amazed1` (astonished, impressed, wow)
- attentive: `attentive1`, `attentive2` (alert, focused, listening)
- calming: `calming1` (soothing, reassuring)
- cheerful: `cheerful1` (happy, upbeat)
- come: `come1` (come here, beckon, invite)
- curious: `curious1`
- enthusiastic: `enthusiastic1`, `enthusiastic2` (excited, eager)
- grateful: `grateful1` (thankful)
- helpful: `helpful1`, `helpful2` (assist)
- laughing: `laughing1`, `laughing2` (laugh, amused)
- loving: `loving1` (love, affectionate)
- proud: `proud1`, `proud2`, `proud3`
- relief: `relief1`, `relief2` (relieved)
- serenity: `serenity1` (peaceful)
- success: `success1`, `success2` (victory, celebrating)
- understanding: `understanding1`, `understanding2` (got it, comprehend)
- welcoming: `welcoming1`, `welcoming2` (welcome, greet)
- yes: `yes1`, `yes_sad1` (agree, affirmative; “sad yes” selects `yes_sad1`)

## Thinking and uncertainty

- anxiety: `anxiety1` (anxious, nervous, worried)
- confused: `confused1` (puzzled)
- incomprehensible: `incomprehensible2` (baffled, don't understand)
- inquiring: `inquiring1`, `inquiring2`, `inquiring3` (questioning)
- lost: `lost1`
- oops: `oops1`, `oops2` (mistake)
- surprised: `surprised1`, `surprised2` (startled)
- thoughtful: `thoughtful1`, `thoughtful2` (thinking, contemplative)
- uncertain: `uncertain1` (unsure, doubtful)

## Low energy and vulnerable

- boredom: `boredom1`, `boredom2` (bored)
- downcast: `downcast1` (dejected)
- exhausted: `exhausted1` (wiped out)
- lonely: `lonely1`
- resigned: `resigned1` (give up, accept defeat)
- sad: `sad1`, `sad2` (unhappy)
- shy: `shy1` (bashful)
- sleep: `sleep1` (sleepy)
- tired: `tired1`
- uncomfortable: `uncomfortable1` (awkward, uneasy)

## Negative and defensive

- contempt: `contempt1` (disdain, dismissive)
- disgusted: `disgusted1` (grossed out)
- displeased: `displeased1`, `displeased2` (dissatisfied)
- fear: `fear1` (afraid)
- frustrated: `frustrated1`
- furious: `furious1` (livid)
- go away: `go_away1` (shoo, leave)
- impatient: `impatient1`, `impatient2`
- indifferent: `indifferent1` (meh, apathetic)
- irritated: `irritated1`, `irritated2` (annoyed)
- no: `no1`, `no_excited1`, `no_sad1` (decline, refuse; “excited no” and “sad no” select their exact variants)
- rage: `rage1` (enraged)
- reprimand: `reprimand1`, `reprimand2`, `reprimand3` (scold, admonish)
- scared: `scared1` (frightened)

## Theatrical and movement reactions

- dance: `dance1`, `dance2`, `dance3` (legacy emotion-dataset motions; explicit dance requests use the separate official dance library)
- dying: `dying1` (dramatic death)
- electric: `electric1` (energized)

## Selection rules

- Exact names win: “express proud three” selects `proud3`.
- Family names choose the first installed variant: “act proud” selects `proud1`.
- Common aliases resolve to a family: “show victory” selects `success1`.
- Authored emotion moves get exclusive body control; face tracking, sound tracking,
  and speech wobble pause until the move completes.
- New upstream recordings appear as unmapped until intentionally assigned to a
  family, preventing silent or nonsensical gesture selection.

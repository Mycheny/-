from PIL import ImageFont
import numpy as np
from OpenGL import GL, GLU


def is_cjk(uchar):
    '''
    Checks for an unicode whether a CJK character
    '''
    #    cjk = (u'\u4e00',u'\u9fa5')
    cjk = (u'\u2e80', u'\ufe4f')
    if cjk[0] <= uchar <= cjk[1]:
        return True
    else:
        return False


def is_ascii(uchar):
    '''
    Checks for an unicode whether a ASCII character
    '''
    ascii = (u'\u0000', u'\u00ff')
    if ascii[0] <= uchar <= ascii[1]:
        return True
    else:
        return False


def is_other(uchar):
    '''
    Checks for an unicode whether an ASCII or CJK character
    '''
    if not (is_cjk(uchar) or is_ascii(uchar)):
        return True
    else:
        return False


def nextpow2(x):
    '''
    If num isn't a power of 2, will return the next higher power of two
    '''
    if x >= 1.0:
        return np.int(2 ** np.ceil(np.log2(x)))
    else:
        print("cannot convert negetive float to integer:", x)


def getCharData(ft, uchar):
    '''
    '''
    # Use our helper function to get the widths of
    # the bitmap data that we will need in order to create
    # our texture.

    if isinstance(uchar, int):
        glyph = ft.getmask(chr(uchar))
    elif isinstance(uchar, str):
        if is_other(uchar):
            return [None] * 5
        else:
            glyph = ft.getmask(uchar)
    elif isinstance(uchar, str):
        uchar = str(uchar)
        if is_other(uchar):
            return [None] * 5
        else:
            glyph = ft.getmask(uchar)
    else:
        return [None] * 5
    glyph_width, glyph_height = glyph.size
    # We are using PIL's wrapping for FreeType. As a result, we don't have
    # direct access to glyph.advance or other attributes, so we add a 1 pixel pad.
    width = nextpow2(glyph_width + 1)
    height = nextpow2(glyph_height + 1)
    # python GL will accept lists of integers or strings, but not Numeric arrays
    # so, we buildup a string for our glyph's texture from the Numeric bitmap

    # Here we fill in the data for the expanded bitmap.
    # Notice that we are using two channel bitmap (one for
    # luminocity and one for alpha), but we assign
    # both luminocity and alpha to the value that we
    # find in the FreeType bitmap.
    # We use the ?: operator so that value which we use
    # will be 0 if we are in the padding zone, and whatever
    # is the the Freetype bitmap otherwise.
    expanded_data = ""
    for j in range(height):
        for i in range(width):
            if (i >= glyph_width) or (j >= glyph_height):
                value = chr(0)
                expanded_data += value
                expanded_data += value
            else:
                value = chr(glyph.getpixel((i, j)))
                expanded_data += value
                expanded_data += value

    return glyph_width, glyph_height, width, height, expanded_data


def make_dlist(ft, ch, list_base, tex_base_list, color=[0, 1, 0]):
    '''
    Given an integer char code, build a GL texture into texture_array,
    build a GL display list for display list number display_list_base + ch.
    Populate the glTexture for the integer ch and construct a display
    list that renders the texture for ch.
    Note, that display_list_base and texture_base are supposed
    to be preallocated for 256 consecutive display lists and and
    array of textures.
    '''
    # Load char data
    glyph_width, glyph_height, width, height, expanded_data = getCharData(ft, ch)
    if not glyph_width:
        return
    # -------------- Build the gl texture ------------

    # Now we just setup some texture paramaters.
    ID = GL.glGenTextures(1)
    tex_base_list[ch] = ID
    GL.glBindTexture(GL.GL_TEXTURE_2D, ID)
    GL.glTexParameterf(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR)
    GL.glTexParameterf(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR)

    border = 0
    # Here we actually create the texture itself, notice
    # that we are using GL_LUMINANCE_ALPHA to indicate that
    # we are using 2 channel data.
    GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, GL.GL_RGBA, width, height, border,
                    GL.GL_LUMINANCE_ALPHA, GL.GL_UNSIGNED_BYTE, expanded_data)

    # With the texture created, we don't need to expanded data anymore
    expanded_data = None

    # --- Build the gl display list that draws the texture for this character ---

    # So now we can create the display list
    GL.glNewList(list_base + ch, GL.GL_COMPILE)

    if ch == ord(" "):
        glyph_advance = glyph_width
        GL.glTranslatef(glyph_advance, 0, 0)
        GL.glEndList()
    else:
        GL.glBindTexture(GL.GL_TEXTURE_2D, ID)
        GL.glPushMatrix()

        # // first we need to move over a little so that
        # // the character has the right amount of space
        # // between it and the one before it.
        # glyph_left = glyph.bbox [0]
        # glTranslatef(glyph_left, 0, 0)

        # // Now we move down a little in the case that the
        # // bitmap extends past the bottom of the line
        # // this is only true for characters like 'g' or 'y'.
        # glyph_descent = glyph.decent
        # glTranslatef(0, glyph_descent, 0)

        # //Now we need to account for the fact that many of
        # //our textures are filled with empty padding space.
        # //We figure what portion of the texture is used by
        # //the actual character and store that information in
        # //the x and y variables, then when we draw the
        # //quad, we will only reference the parts of the texture
        # //that we contain the character itself.
        x = np.float(glyph_width) / np.float(width)
        y = np.float(glyph_height) / np.float(height)

        # //Here we draw the texturemaped quads.
        # //The bitmap that we got from FreeType was not
        # //oriented quite like we would like it to be,
        # //so we need to link the texture to the quad
        # //so that the result will be properly aligned.
        GL.glBegin(GL.GL_QUADS)
        GL.glColor3fv(color)
        GL.glTexCoord2f(0, 0), GL.glVertex2f(0, glyph_height)
        GL.glTexCoord2f(0, y), GL.glVertex2f(0, 0)
        GL.glTexCoord2f(x, y), GL.glVertex2f(glyph_width, 0)
        GL.glTexCoord2f(x, 0), GL.glVertex2f(glyph_width, glyph_height)
        GL.glEnd()
        GL.glPopMatrix()

        # Note, PIL's FreeType interface hides the advance from us.
        # Normal PIL clients are rendering an entire string through FreeType, not
        # a single character at a time like we are doing here.
        # Because the advance value is hidden from we will advance
        # the "pen" based upon the rendered glyph's width. This is imperfect.
        GL.glTranslatef(glyph_width + 0.75, 0, 0)

        # //increment the raster position as if we were a bitmap font.
        # //(only needed if you want to calculate text length)
        # //glBitmap(0,0,0,0,face->glyph->advance.x >> 6,0,NULL)

        # //Finnish the display list
        GL.glEndList()
    return


def dispCJK(ft, uchar, tex_base_list, color=[1, 1, 0]):
    '''
    '''
    # Load char data
    glyph_width, glyph_height, width, height, expanded_data = getCharData(ft, uchar)
    if glyph_width == None:
        return
        # -------------- Build the gl texture ------------

    # Now we just setup some texture paramaters.
    ID = GL.glGenTextures(1)
    tex_base_list.append(ID)
    GL.glBindTexture(GL.GL_TEXTURE_2D, ID)
    GL.glTexParameterf(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR)
    GL.glTexParameterf(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR)

    border = 0
    # Here we actually create the texture itself, notice
    # that we are using GL_LUMINANCE_ALPHA to indicate that
    # we are using 2 channel data.
    GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, GL.GL_RGBA, width, height, border,
                    GL.GL_LUMINANCE_ALPHA, GL.GL_UNSIGNED_BYTE, expanded_data)

    # With the texture created, we don't need to expanded data anymore
    expanded_data = None
    GL.glBindTexture(GL.GL_TEXTURE_2D, ID)
    GL.glPushMatrix()
    x = np.float(glyph_width) / np.float(width)
    y = np.float(glyph_height) / np.float(height)
    GL.glBegin(GL.GL_QUADS)
    GL.glColor3fv(color)
    GL.glTexCoord2f(0, 0), GL.glVertex2f(0, glyph_height)
    GL.glTexCoord2f(0, y), GL.glVertex2f(0, 0)
    GL.glTexCoord2f(x, y), GL.glVertex2f(glyph_width, 0)
    GL.glTexCoord2f(x, 0), GL.glVertex2f(glyph_width, glyph_height)
    GL.glEnd()
    GL.glPopMatrix()
    GL.glTranslatef(glyph_width + 0.75, 0, 0)
    return


def pushScreenCoordinateMatrix():
    # A fairly straight forward function that pushes
    # a projection matrix that will make object world
    # coordinates identical to window coordinates.
    GL.glPushAttrib(GL.GL_TRANSFORM_BIT)
    viewport = GL.glGetIntegerv(GL.GL_VIEWPORT)
    GL.glMatrixMode(GL.GL_PROJECTION)
    GL.glPushMatrix()
    GL.glLoadIdentity()
    GLU.gluOrtho2D(viewport[0], viewport[2], viewport[1], viewport[3])
    GL.glPopAttrib()
    return


def pop_projection_matrix():
    # Pops the projection matrix without changing the current
    # MatrixMode.
    GL.glPushAttrib(GL.GL_TRANSFORM_BIT)
    GL.glMatrixMode(GL.GL_PROJECTION)
    GL.glPopMatrix()
    GL.glPopAttrib()
    return


class font_data(object):
    '''
    '''

    def __init__(self, facename, pixel_height):
        # We haven't yet allocated textures or display lists
        self.m_allocated = False
        self.m_font_height = pixel_height
        self.m_facename = facename

        # Try to obtain the FreeType font
        try:
            self.ft = ImageFont.truetype(facename, pixel_height)
        except:
            raise ValueError("Unable to locate true type font '%s'" % (facename))
        # Here we ask opengl to allocate resources for
        # all the textures and displays lists which we
        # are about to create.
        # Note: only ASCII character
        n = 256
        self.m_list_base = GL.glGenLists(n)

        # Consturct a list of 256 elements. This
        # list will be assigned the texture IDs we create for each glyph
        self.textures = [None] * n
        self.cjk_textures = []
        # This is where we actually create each of the fonts display lists.
        for i in range(n):
            make_dlist(self.ft, i, self.m_list_base, self.textures)

        self.m_allocated = True

    def glPrint(self, x, y, string, color=[1, 0, 0]):
        '''
        '''
        # We want a coordinate system where things coresponding to window pixels.

        pushScreenCoordinateMatrix()
        # //We make the height about 1.5* that of
        h = np.float(self.m_font_height) / 0.63

        if not string:
            pop_projection_matrix()
            return
        else:
            if not isinstance(string, str):
                try:
                    string = str(string)
                except:
                    raise ValueError("Can not convert to unicode").with_traceback(string)
            # //Here is some code to split the text that we have been
            # //given into a set of lines.
            # //This could be made much neater by using
            # //a regular expression library such as the one avliable from
            # //boost.org (I've only done it out by hand to avoid complicating
            # //this tutorial with unnecessary library dependencies).
            # //Note: python string object has convenience method for this :)
            lines = string.split("\n")
            GL.glPushAttrib(GL.GL_LIST_BIT | GL.GL_CURRENT_BIT | GL.GL_ENABLE_BIT | GL.GL_TRANSFORM_BIT)
            GL.glMatrixMode(GL.GL_MODELVIEW)
            #            GL.glDisable(GL.GL_LIGHTING)
            #            GL.glEnable(GL.GL_TEXTURE_2D)
            #            GL.glDisable(GL.GL_DEPTH_TEST)
            #            GL.glEnable(GL.GL_BLEND)
            #            GL.glBlendFunc(GL.GL_SRC_ALPHA,GL.GL_ONE_MINUS_SRC_ALPHA)

            GL.glListBase(self.m_list_base)
            modelview_matrix = GL.glGetFloatv(GL.GL_MODELVIEW_MATRIX)

            # //This is where the text display actually happens.
            # //For each line of text we reset the modelview matrix
            # //so that the line's text will start in the correct position.
            # //Notice that we need to reset the matrix, rather than just translating
            # //down by h. This is because when each character is
            # //draw it modifies the current matrix so that the next character
            # //will be drawn immediatly after it.
            for i in range(len(lines)):
                line = lines[i]
                GL.glPushMatrix()
                GL.glLoadIdentity()
                GL.glTranslatef(x, y - h * i, 0);
                GL.glMultMatrixf(modelview_matrix);

                # //  The commented out raster position stuff can be useful if you need to
                # //  know the length of the text that you are creating.
                # //  If you decide to use it make sure to also uncomment the glBitmap command
                # //  in make_dlist().
                # //    glRasterPos2f(0,0);
                #            glCallLists (line)
                for tt in line:
                    if is_cjk(tt):
                        dispCJK(self.ft, tt, self.cjk_textures)
                    else:
                        #            print 'ascii',tt
                        GL.glCallList(ord(tt) + 1)
                # //    rpos = glGetFloatv (GL_CURRENT_RASTER_POSITION)
                # //    float len=x-rpos[0];
                GL.glPopMatrix()
            GL.glPopAttrib()
            pop_projection_matrix()

    def release(self):
        """ Release the gl resources for this Face.
            (This provides the functionality of KillFont () and font_data::clean ()
        """
        if self.m_allocated:
            # Free up the glTextures and the display lists for our face
            GL.glDeleteLists(self.m_list_base, 256);
            for ID in self.textures:
                GL.glDeleteTextures(ID)
            if self.cjk_textures:
                for ID in self.cjk_textures:
                    GL.glDeleteTextures(ID)
            # Extra defensive. Clients that continue to try and use this object
            # will now trigger exceptions.
            self.list_base = None
            self.m_allocated = False

    def __del__(self):
        """ Python destructor for when no more refs to this Face object """
        self.release()


# Unit Test harness if this python module is run directly.
if __name__ == "__main__":
    print("testing availability of freetype font arial\n")
    ft = ImageFont.truetype("msyh.ttc", 15)
    if ft:
        print("Found the TrueType font 'Test.ttf'")
    else:
        print("faild to find the TrueTYpe font 'arial'\n")